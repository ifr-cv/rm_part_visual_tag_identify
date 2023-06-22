/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

License Agreement
For Open Source Computer Vision Library
(3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "visual_tag.cpu.hpp"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <utility>

namespace cv {
    namespace cpu {
        namespace visual_tag {
            using namespace std;

            ///标记颜色, 仅红/蓝
            enum MarkerColor {
                RED, BLUE
            };

            /**
            * @brief 将 3 通道的输入图像转换为灰色, 其它则复制
            */
            static void _convertToGrey(InputArray _in, OutputArray _out) {

                CV_Assert(_in.getMat().channels() == 1 || _in.getMat().channels() == 3);

                _out.create(_in.getMat().size(), CV_8UC1);
                if (_in.getMat().type() == CV_8UC3)
                    cvtColor(_in.getMat(), _out.getMat(), COLOR_BGR2GRAY);
                else
                    _in.getMat().copyTo(_out);
            }


            /**
            * @brief Given a tresholded image, find the contours, calculate their polygonal approximation and take those that accomplish some conditions
            */
            static void _findMarkerContours(const Mat &contoursImg, vector<vector<Point2f> > &candidates,
                                            vector<vector<Point> > &contoursOut, vector<MarkerColor> &colorsOut,
                                            const MarkerColor &color,
                                            double minPerimeterRate, double maxPerimeterRate, double accuracyRate,
                                            double minCornerDistanceRate, int minDistanceToBorder) {

                CV_Assert(minPerimeterRate > 0 && maxPerimeterRate > 0 && accuracyRate > 0 && minCornerDistanceRate >= 0 && minDistanceToBorder >= 0);

                // calculate maximum and minimum sizes in pixels
                auto minPerimeterPixels = (unsigned int) (minPerimeterRate * max(contoursImg.cols, contoursImg.rows));
                auto maxPerimeterPixels = (unsigned int) (maxPerimeterRate * max(contoursImg.cols, contoursImg.rows));

                vector<vector<Point> > contours;
                findContours(contoursImg, contours, RETR_LIST, CHAIN_APPROX_NONE);
                // now filter list of contours
                for (const auto &contour: contours) {
                    // check perimeter
                    if (contour.size() < minPerimeterPixels || contour.size() > maxPerimeterPixels)
                        continue;

                    // check is square and is convex
                    vector<Point> approxCurve;
                    approxPolyDP(contour, approxCurve, double(contour.size()) * accuracyRate, true);
                    if (approxCurve.size() != 4 || !isContourConvex(approxCurve)) continue;

                    // check min distance between corners
                    double minDistSq = max(contoursImg.cols, contoursImg.rows) * max(contoursImg.cols, contoursImg.rows);
                    for (int j = 0; j < 4; j++) {
                        double d = (double) (approxCurve[j].x - approxCurve[(j + 1) % 4].x) *
                                   (double) (approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                                   (double) (approxCurve[j].y - approxCurve[(j + 1) % 4].y) *
                                   (double) (approxCurve[j].y - approxCurve[(j + 1) % 4].y);
                        minDistSq = min(minDistSq, d);
                    }
                    double minCornerDistancePixels = double(contour.size()) * minCornerDistanceRate;
                    if (minDistSq < minCornerDistancePixels * minCornerDistancePixels) continue;

                    // check if it is too near to the image border
                    bool tooNearBorder = false;
                    for (int j = 0; j < 4; j++) {
                        if (approxCurve[j].x < minDistanceToBorder || approxCurve[j].y < minDistanceToBorder ||
                            approxCurve[j].x > contoursImg.cols - 1 - minDistanceToBorder ||
                            approxCurve[j].y > contoursImg.rows - 1 - minDistanceToBorder)
                            tooNearBorder = true;
                    }
                    if (tooNearBorder) continue;

                    // if it passes all the test, add to candidates vector
                    vector<Point2f> currentCandidate;
                    currentCandidate.resize(4);
                    for (int j = 0; j < 4; j++) {
                        currentCandidate[j] = Point2f((float) approxCurve[j].x, (float) approxCurve[j].y);
                    }
                    candidates.push_back(currentCandidate);
                    contoursOut.push_back(contour);
                    colorsOut.push_back(color);
                }
            }


            /**
            * @brief Assure order of candidate corners is clockwise direction
            */
            static void _reorderCandidatesCorners(vector<vector<Point2f> > &candidates) {

                for (auto &candidate: candidates) {
                    double dx1 = candidate[1].x - candidate[0].x;
                    double dy1 = candidate[1].y - candidate[0].y;
                    double dx2 = candidate[2].x - candidate[0].x;
                    double dy2 = candidate[2].y - candidate[0].y;
                    double crossProduct = (dx1 * dy2) - (dy1 * dx2);

                    if (crossProduct < 0.0) { // not clockwise direction
                        swap(candidate[1], candidate[3]);
                    }
                }
            }


            /**
            * @brief Check candidates that are too close to each other and remove the smaller one
            */
            static void _filterTooCloseCandidates(const vector<vector<Point2f> > &candidatesIn,
                                                  vector<vector<Point2f> > &candidatesOut,
                                                  const vector<vector<Point> > &contoursIn,
                                                  vector<vector<Point> > &contoursOut,
                                                  const vector<MarkerColor> &colorsIn,
                                                  vector<MarkerColor> &colorsOut,
                                                  double minMarkerDistanceRate) {

                CV_Assert(minMarkerDistanceRate >= 0);

                vector<pair<int, int> > nearCandidates;
                for (unsigned int i = 0; i < candidatesIn.size(); i++) {
                    for (unsigned int j = i + 1; j < candidatesIn.size(); j++) {

                        int minimumPerimeter = min((int) contoursIn[i].size(), (int) contoursIn[j].size());

                        // fc is the first corner considered on one of the markers, 4 combinatios are posible
                        for (int fc = 0; fc < 4; fc++) {
                            double distSq = 0;
                            for (int c = 0; c < 4; c++) {
                                // modC is the corner considering first corner is fc
                                int modC = (c + fc) % 4;
                                distSq += (candidatesIn[i][modC].x - candidatesIn[j][c].x) *
                                          (candidatesIn[i][modC].x - candidatesIn[j][c].x) +
                                          (candidatesIn[i][modC].y - candidatesIn[j][c].y) *
                                          (candidatesIn[i][modC].y - candidatesIn[j][c].y);
                            }
                            distSq /= 4.;

                            // if mean square distance is too low, remove the smaller one of the two markers
                            double minMarkerDistancePixels = double(minimumPerimeter) * minMarkerDistanceRate;
                            if (distSq < minMarkerDistancePixels * minMarkerDistancePixels) {
                                nearCandidates.emplace_back(i, j);
                                break;
                            }
                        }
                    }
                }

                // mark smaller one in pairs to remove
                vector<bool> toRemove(candidatesIn.size(), false);
                for (auto &nearCandidate: nearCandidates) {
                    // if one of the marker has been already markerd to removed, dont need to do anything
                    if (toRemove[nearCandidate.first] || toRemove[nearCandidate.second]) continue;
                    size_t perimeter1 = contoursIn[nearCandidate.first].size();
                    size_t perimeter2 = contoursIn[nearCandidate.second].size();
                    if (perimeter1 > perimeter2)
                        toRemove[nearCandidate.second] = true;
                    else
                        toRemove[nearCandidate.first] = true;
                }

                // remove extra candidates
                candidatesOut.clear();
                int totalRemaining = 0;
                for (auto &&i: toRemove) if (!i) totalRemaining++;
                candidatesOut.resize(totalRemaining);
                contoursOut.resize(totalRemaining);
                colorsOut.resize(totalRemaining);
                for (unsigned int i = 0, currIdx = 0; i < candidatesIn.size(); i++) {
                    if (toRemove[i]) continue;
                    candidatesOut[currIdx] = candidatesIn[i];
                    contoursOut[currIdx] = contoursIn[i];
                    colorsOut[currIdx] = colorsIn[i];
                    currIdx++;
                }
            }


            /**
            * ParallelLoopBody 类
            * 用于使用不同的阈值窗口大小对基本候选检测进行并行化。从函数 _detectInitialCandidates() 调用
            */
            class DetectInitialCandidatesParallel : public ParallelLoopBody {
            public:
                DetectInitialCandidatesParallel(const Mat *_rgb,
                                                vector<vector<vector<Point2f> > > *_candidatesArrays,
                                                vector<vector<vector<Point> > > *_contoursArrays,
                                                vector<vector<MarkerColor> > *_colorsArrays,
                                                DetectorParameters *_params)
                        : rgb(_rgb), candidatesArrays(_candidatesArrays),
                          contoursArrays(_contoursArrays), colorsArrays(_colorsArrays),
                          params(_params) {}

                void operator()(const Range &range) const override {
                    const int begin = range.start;
                    const int end = range.end;

                    for (int i = begin; i < end; i++) {

                        // threshold
                        Mat thresh;
                        cv::cvtColor(*rgb, thresh, i ? cv::ColorConversionCodes::COLOR_BGR2HSV : cv::ColorConversionCodes::COLOR_RGB2HSV);
                        static const cv::Scalar lower_blue(110, 100, 10);
                        static const cv::Scalar upper_blue(130, 255, 255);
                        cv::inRange(thresh, lower_blue, upper_blue, thresh);

                        // DEBUG show
//                    cv::imshow(i == MarkerColor::BLUE ? "thresh-b" : "thresh-r", thresh);

                        // detect rectangles
                        _findMarkerContours(thresh, (*candidatesArrays)[i], (*contoursArrays)[i], (*colorsArrays)[i],
                                            static_cast<MarkerColor>(i),
                                            params->minMarkerPerimeterRate, params->maxMarkerPerimeterRate,
                                            params->polygonalApproxAccuracyRate, params->minCornerDistanceRate,
                                            params->minDistanceToBorder);
                    }
                }

                DetectInitialCandidatesParallel &operator=(const DetectInitialCandidatesParallel &) = delete;

            private:

                const Mat *rgb;
                vector<vector<vector<Point2f> > > *candidatesArrays;
                vector<vector<vector<Point> > > *contoursArrays;
                vector<vector<MarkerColor> > *colorsArrays;
                DetectorParameters *params;
            };


            /**
            * @brief Initial steps on finding square candidates
            */
            static void _detectInitialCandidates(const Mat &rgb_or_bgr, vector<vector<Point2f> > &candidates,
                                                 vector<vector<Point> > &contours, vector<MarkerColor> &colors,
                                                 DetectorParameters params) {

                /// 阈值数量(仅红&蓝)
                static constexpr int nScales = 2;

                vector<vector<vector<Point2f> > > candidatesArrays(nScales);
                vector<vector<vector<Point> > > contoursArrays(nScales);
                vector<vector<MarkerColor> > colorsArrays(nScales);


                parallel_for_(Range(0, nScales), DetectInitialCandidatesParallel(&rgb_or_bgr, &candidatesArrays,
                                                                                 &contoursArrays, &colorsArrays,
                                                                                 &params));
//            DetectInitialCandidatesParallel(&rgb_or_bgr, &candidatesArrays,
//                                            &contoursArrays, &colorsArrays,
//                                            &params)(Range(0, nScales));

                // join candidates
                for (const auto &data: candidatesArrays)candidates.insert(candidates.end(), data.begin(), data.end());
                for (const auto &data: contoursArrays)contours.insert(contours.end(), data.begin(), data.end());
                for (const auto &data: colorsArrays)colors.insert(colors.end(), data.begin(), data.end());
            }


            /**
            * @brief Detect square candidates in the input image
             * @param image RGB或BGR图像
            */
            static void _detectCandidates(const cv::Mat &image, vector<vector<Point2f> > &_candidates,
                                          vector<vector<Point> > &_contours, vector<MarkerColor> &_colors,
                                          DetectorParameters params) {

                CV_Assert(image.total() != 0);

                vector<vector<Point2f> > candidates;
                vector<vector<Point> > contours;
                vector<MarkerColor> colors;
                /// 2. DETECT FIRST SET OF CANDIDATES
                _detectInitialCandidates(image, candidates, contours, colors, params);

                /// 3. SORT CORNERS
                _reorderCandidatesCorners(candidates);

                /// 4. FILTER OUT NEAR CANDIDATE PAIRS
                _filterTooCloseCandidates(candidates, _candidates, contours, _contours, colors, _colors,
                                          params.minMarkerDistanceRate);
            }


            /**
            * @brief Given an input image and candidate corners, extract the bits of the candidate, including
            * the border bits
            */
            static Mat _extractBits(InputArray _image, InputArray _corners, int markerSize,
                                    int markerBorderBits, int cellSize, double cellMarginRate,
                                    double minStdDevOtsu) {

                CV_Assert(_image.getMat().channels() == 1);
                CV_Assert(_corners.total() == 4);
                CV_Assert(markerBorderBits > 0 && cellSize > 0 && cellMarginRate >= 0 && cellMarginRate <= 1);
                CV_Assert(minStdDevOtsu >= 0);

                // number of bits in the marker
                int markerSizeWithBorders = markerSize + 2 * markerBorderBits;
                int cellMarginPixels = int(cellMarginRate * cellSize);

                Mat resultImg; // marker image after removing perspective
                int resultImgSize = markerSizeWithBorders * cellSize;
                Mat resultImgCorners(4, 1, CV_32FC2);
                resultImgCorners.ptr<Point2f>(0)[0] = Point2f(0, 0);
                resultImgCorners.ptr<Point2f>(0)[1] = Point2f((float) resultImgSize - 1, 0);
                resultImgCorners.ptr<Point2f>(0)[2] =
                        Point2f((float) resultImgSize - 1, (float) resultImgSize - 1);
                resultImgCorners.ptr<Point2f>(0)[3] = Point2f(0, (float) resultImgSize - 1);

                // remove perspective
                Mat transformation = getPerspectiveTransform(_corners, resultImgCorners);
                warpPerspective(_image, resultImg, transformation, Size(resultImgSize, resultImgSize),
                                INTER_NEAREST);

                // output image containing the bits
                Mat bits(markerSizeWithBorders, markerSizeWithBorders, CV_8UC1, Scalar::all(0));

                // check if standard deviation is enough to apply Otsu
                // if not enough, it probably means all bits are the same color (black or white)
                Mat mean, stddev;
                // Remove some border just to avoid border noise from perspective transformation
                Mat innerRegion = resultImg.colRange(cellSize / 2, resultImg.cols - cellSize / 2)
                        .rowRange(cellSize / 2, resultImg.rows - cellSize / 2);
                meanStdDev(innerRegion, mean, stddev);
                if (stddev.ptr<double>(0)[0] < minStdDevOtsu) {
                    // all black or all white, depending on mean value
                    if (mean.ptr<double>(0)[0] > 127)
                        bits.setTo(1);
                    else
                        bits.setTo(0);
                    return bits;
                }

                // now extract code, first threshold using Otsu
                threshold(resultImg, resultImg, 125, 255, THRESH_BINARY | THRESH_OTSU);

                // for each cell
                for (int y = 0; y < markerSizeWithBorders; y++) {
                    for (int x = 0; x < markerSizeWithBorders; x++) {
                        int Xstart = x * (cellSize) + cellMarginPixels;
                        int Ystart = y * (cellSize) + cellMarginPixels;
                        Mat square = resultImg(Rect(Xstart, Ystart, cellSize - 2 * cellMarginPixels,
                                                    cellSize - 2 * cellMarginPixels));
                        // count white pixels on each cell to assign its value
                        unsigned int nZ = countNonZero(square);
                        if (nZ > square.total() / 2) bits.at<unsigned char>(y, x) = 1;
                    }
                }

                return bits;
            }


            /**
            * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
            */
            static int _getBorderErrors(const Mat &bits, int markerSize, int borderSize) {

                int sizeWithBorders = markerSize + 2 * borderSize;

                CV_Assert(markerSize > 0 && bits.cols == sizeWithBorders && bits.rows == sizeWithBorders);

                int totalErrors = 0;
                for (int y = 0; y < sizeWithBorders; y++) {
                    for (int k = 0; k < borderSize; k++) {
                        if (bits.ptr<unsigned char>(y)[k] != 0) totalErrors++;
                        if (bits.ptr<unsigned char>(y)[sizeWithBorders - 1 - k] != 0) totalErrors++;
                    }
                }
                for (int x = borderSize; x < sizeWithBorders - borderSize; x++) {
                    for (int k = 0; k < borderSize; k++) {
                        if (bits.ptr<unsigned char>(k)[x] != 0) totalErrors++;
                        if (bits.ptr<unsigned char>(sizeWithBorders - 1 - k)[x] != 0) totalErrors++;
                    }
                }
                return totalErrors;
            }


            /**
            * @brief Tries to identify one candidate given the dictionary
            */
            static bool _identifyOneCandidate(const Dictionary &dictionary, InputArray _image,
                                              InputOutputArray _corners, MarkerColor color, int &idx, DetectorParameters params) {

                CV_Assert(_corners.total() == 4);
                CV_Assert(_image.getMat().total() != 0);
                CV_Assert(params.markerBorderBits > 0);

                // get bits
                Mat candidateBits =
                        _extractBits(_image, _corners, dictionary.markerSize, params.markerBorderBits,
                                     params.perspectiveRemovePixelPerCell,
                                     params.perspectiveRemoveIgnoredMarginPerCell, params.minOtsuStdDev);

                // analyze border bits
                int maximumErrorsInBorder = int(dictionary.markerSize * dictionary.markerSize * params.maxErroneousBitsInBorderRate);
                int borderErrors = _getBorderErrors(candidateBits, dictionary.markerSize, params.markerBorderBits);
                if (borderErrors > maximumErrorsInBorder) return false; // border is wrong

                // take only inner bits
                Mat onlyBits(candidateBits,
                             Range(params.markerBorderBits, candidateBits.rows - params.markerBorderBits),
                             Range(params.markerBorderBits, candidateBits.rows - params.markerBorderBits));

                // try to indentify the marker
                int rotation;
                if (!dictionary.identify(onlyBits, idx, rotation, params.errorCorrectionRate))
                    return false;
                else {
                    // Additional color id
                    idx += dictionary.bytesList.rows * color;
                    // shift corner positions to the correct rotation
                    if (rotation != 0) {
                        Mat copyPoints = _corners.getMat().clone();
                        for (int j = 0; j < 4; j++)
                            _corners.getMat().ptr<Point2f>(0)[j] =
                                    copyPoints.ptr<Point2f>(0)[(j + 4 - rotation) % 4];
                    }
                    return true;
                }
            }


            /**
            * ParallelLoopBody class for the parallelization of the marker identification step
            * Called from function _identifyCandidates()
            */
            class IdentifyCandidatesParallel : public ParallelLoopBody {
            public:
                IdentifyCandidatesParallel(const Mat *_grey, InputArrayOfArrays _candidates,
                                           InputArrayOfArrays _contours, const vector<MarkerColor> *_colors,
                                           const Dictionary *_dictionary,
                                           vector<int> *_idsTmp, vector<char> *_validCandidates,
                                           DetectorParameters *_params)
                        : grey(_grey), candidates(_candidates), contours(_contours), colors(_colors), dictionary(_dictionary),
                          idsTmp(_idsTmp), validCandidates(_validCandidates), params(_params) {}

                void operator()(const Range &range) const override {
                    const int begin = range.start;
                    const int end = range.end;

                    for (int i = begin; i < end; i++) {
                        int currId;
                        Mat currentCandidate = candidates.getMat(i);
                        const MarkerColor &color = colors->at(i);
                        if (_identifyOneCandidate(*dictionary, *grey, currentCandidate, color, currId, *params)) {
                            (*validCandidates)[i] = 1;
                            (*idsTmp)[i] = currId;
                        }
                    }
                }

                IdentifyCandidatesParallel &operator=(const IdentifyCandidatesParallel &) = delete; // to quiet MSVC

            private:

                const Mat *grey;
                InputArrayOfArrays candidates, contours;
                const vector<MarkerColor> *colors;
                const Dictionary *dictionary;
                vector<int> *idsTmp;
                vector<char> *validCandidates;
                DetectorParameters *params;
            };


            /**
            * @brief Identify square candidates according to a marker dictionary
            */
            static void _identifyCandidates(InputArray _image, InputArrayOfArrays _candidates,
                                            InputArrayOfArrays _contours, const vector<MarkerColor> &_colors,
                                            const Dictionary &dictionary, OutputArrayOfArrays _accepted,
                                            OutputArray _ids, DetectorParameters params,
                                            OutputArrayOfArrays _rejected = noArray()) {

                int ncandidates = (int) _candidates.total();

                vector<Mat> accepted;
                vector<Mat> rejected;
                vector<int> ids;

                CV_Assert(_image.getMat().total() != 0);

                Mat grey;
                _convertToGrey(_image.getMat(), grey);

                vector<int> idsTmp(ncandidates, -1);
                vector<char> validCandidates(ncandidates, 0);

                //// Analyze each of the candidates
                // for (int i = 0; i < ncandidates; i++) {
                //    int currId = i;
                //    GpuMat currentCandidate = _candidates.getMat(i);
                //    if (_identifyOneCandidate(dictionary, grey, currentCandidate, currId, params)) {
                //        validCandidates[i] = 1;
                //        idsTmp[i] = currId;
                //    }
                //}

                // this is the parallel call for the previous commented loop (result is equivalent)
                parallel_for_(Range(0, ncandidates),
                              IdentifyCandidatesParallel(&grey, _candidates, _contours, &_colors, &dictionary, &idsTmp,
                                                         &validCandidates, &params));

                for (int i = 0; i < ncandidates; i++) {
                    if (validCandidates[i] == 1) {
                        accepted.push_back(_candidates.getMat(i));
                        ids.push_back(idsTmp[i]);
                    } else {
                        rejected.push_back(_candidates.getMat(i));
                    }
                }

                // parse output
                _accepted.create((int) accepted.size(), 1, CV_32FC2);
                for (unsigned int i = 0; i < accepted.size(); i++) {
                    _accepted.create(4, 1, CV_32FC2, int(i), true);
                    Mat m = _accepted.getMat(int(i));
                    accepted[i].copyTo(m);
                }

                _ids.create((int) ids.size(), 1, CV_32SC1);
                for (unsigned int i = 0; i < ids.size(); i++)
                    _ids.getMat().ptr<int>(0)[i] = ids[i];

                if (_rejected.needed()) {
                    _rejected.create((int) rejected.size(), 1, CV_32FC2);
                    for (unsigned int i = 0; i < rejected.size(); i++) {
                        _rejected.create(4, 1, CV_32FC2, int(i), true);
                        Mat m = _rejected.getMat(int(i));
                        rejected[i].copyTo(m);
                    }
                }
            }


            /**
            * @brief Final filter of markers after its identification
            */
            static void _filterDetectedMarkers(InputArrayOfArrays _inCorners, InputArray _inIds,
                                               OutputArrayOfArrays _outCorners, OutputArray _outIds) {

                CV_Assert(_inCorners.total() == _inIds.total());
                if (_inCorners.total() == 0) return;

                // mark markers that will be removed
                vector<bool> toRemove(_inCorners.total(), false);
                bool atLeastOneRemove = false;

                // remove repeated markers with same id, if one contains the other (doble border bug)
                for (unsigned int i = 0; i < _inCorners.total() - 1; i++) {
                    for (unsigned int j = i + 1; j < _inCorners.total(); j++) {
                        if (_inIds.getMat().ptr<int>(0)[i] != _inIds.getMat().ptr<int>(0)[j]) continue;

                        // check if first marker is inside second
                        bool inside = true;
                        for (unsigned int p = 0; p < 4; p++) {
                            Point2f point = _inCorners.getMat(int(j)).ptr<Point2f>(0)[p];
                            if (pointPolygonTest(_inCorners.getMat(int(i)), point, false) < 0) {
                                inside = false;
                                break;
                            }
                        }
                        if (inside) {
                            toRemove[j] = true;
                            atLeastOneRemove = true;
                            continue;
                        }

                        // check the second marker
                        inside = true;
                        for (unsigned int p = 0; p < 4; p++) {
                            Point2f point = _inCorners.getMat(int(i)).ptr<Point2f>(0)[p];
                            if (pointPolygonTest(_inCorners.getMat(int(j)), point, false) < 0) {
                                inside = false;
                                break;
                            }
                        }
                        if (inside) {
                            toRemove[i] = true;
                            atLeastOneRemove = true;
                            continue;
                        }
                    }
                }

                // parse output
                if (atLeastOneRemove) {
                    vector<Mat> filteredCorners;
                    vector<int> filteredIds;

                    for (unsigned int i = 0; i < toRemove.size(); i++) {
                        if (!toRemove[i]) {
                            filteredCorners.push_back(_inCorners.getMat(int(i)).clone());
                            filteredIds.push_back(_inIds.getMat().ptr<int>(0)[i]);
                        }
                    }

                    _outIds.create((int) filteredIds.size(), 1, CV_32SC1);
                    for (unsigned int i = 0; i < filteredIds.size(); i++)
                        _outIds.getMat().ptr<int>(0)[i] = filteredIds[i];

                    _outCorners.create((int) filteredCorners.size(), 1, CV_32FC2);
                    for (unsigned int i = 0; i < filteredCorners.size(); i++) {
                        _outCorners.create(4, 1, CV_32FC2, int(i), true);
                        filteredCorners[i].copyTo(_outCorners.getMat(int(i)));
                    }
                }
            }


            /**
            * @brief Return object points for the system centered in a single marker, given the marker length
            */
            static void _getSingleMarkerObjectPoints(float markerLength, OutputArray _objPoints) {

                CV_Assert(markerLength > 0);

                _objPoints.create(4, 1, CV_32FC3);
                Mat objPoints = _objPoints.getMat();
                // set coordinate system in the middle of the marker, with Z pointing out
                objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
                objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
                objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
                objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);
            }


            /**
            * ParallelLoopBody class for the parallelization of the marker corner subpixel refinement
            * Called from function detectMarkers()
            */
            class MarkerSubpixelParallel : public ParallelLoopBody {
            public:
                MarkerSubpixelParallel(const Mat *_grey, OutputArrayOfArrays _corners,
                                       DetectorParameters *_params)
                        : grey(_grey), corners(_corners), params(_params) {}

                void operator()(const Range &range) const override {
                    const int begin = range.start;
                    const int end = range.end;

                    for (int i = begin; i < end; i++) {
                        cornerSubPix(*grey, corners.getMat(i),
                                     Size(params->cornerRefinementWinSize, params->cornerRefinementWinSize),
                                     Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                                                params->cornerRefinementMaxIterations,
                                                                params->cornerRefinementMinAccuracy));
                    }
                }

                MarkerSubpixelParallel &operator=(const MarkerSubpixelParallel &) = delete; // to quiet MSVC
            private:

                const Mat *grey;
                OutputArrayOfArrays corners;
                DetectorParameters *params;
            };


            void detectMarkers(InputArray _image, const Dictionary &dictionary, OutputArrayOfArrays _corners,
                               OutputArray _ids, DetectorParameters params,
                               OutputArrayOfArrays _rejectedImgPoints) {

                CV_Assert(_image.getMat().total() != 0);

                Mat grey;
                _convertToGrey(_image.getMat(), grey);

                /// STEP 1: Detect marker candidates
                vector<vector<Point2f> > candidates;
                vector<vector<Point> > contours;
                vector<MarkerColor> colors;
                _detectCandidates(_image.getMat(), candidates, contours, colors, params);

                /// STEP 2: Check candidate codification (identify markers)
                _identifyCandidates(grey, candidates, contours, colors, dictionary, _corners, _ids, params,
                                    _rejectedImgPoints);

                /// STEP 3: Filter detected markers;
                _filterDetectedMarkers(_corners, _ids, _corners, _ids);

                /// STEP 4: Corner refinement
                if (params.doCornerRefinement) {
                    CV_Assert(params.cornerRefinementWinSize > 0 && params.cornerRefinementMaxIterations > 0 &&
                              params.cornerRefinementMinAccuracy > 0);

                    //// do corner refinement for each of the detected markers
                    // for (unsigned int i = 0; i < _corners.total(); i++) {
                    //    cornerSubPix(grey, _corners.getMat(i),
                    //                 Size(params.cornerRefinementWinSize, params.cornerRefinementWinSize),
                    //                 Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                    //                                            params.cornerRefinementMaxIterations,
                    //                                            params.cornerRefinementMinAccuracy));
                    //}

                    // this is the parallel call for the previous commented loop (result is equivalent)
                    parallel_for_(Range(0, (int) _corners.total()),
                                  MarkerSubpixelParallel(&grey, _corners, &params));
                }
            }


            /**
            * ParallelLoopBody class for the parallelization of the single markers pose estimation
            * Called from function estimatePoseSingleMarkers()
            */
            class SinglePoseEstimationParallel : public ParallelLoopBody {
            public:
                SinglePoseEstimationParallel(Mat &_markerObjPoints, InputArrayOfArrays _corners,
                                             InputArray _cameraMatrix, InputArray _distCoeffs,
                                             Mat &_rvecs, Mat &_tvecs)
                        : markerObjPoints(_markerObjPoints), corners(_corners), cameraMatrix(_cameraMatrix),
                          distCoeffs(_distCoeffs), rvecs(_rvecs), tvecs(_tvecs) {}

                void operator()(const Range &range) const override {
                    const int begin = range.start;
                    const int end = range.end;

                    for (int i = begin; i < end; i++) {
                        solvePnP(markerObjPoints, corners.getMat(i), cameraMatrix, distCoeffs,
                                 rvecs.at<Vec3d>(0, i), tvecs.at<Vec3d>(0, i));
                    }
                }

                SinglePoseEstimationParallel &operator=(const SinglePoseEstimationParallel &) = delete; // to quiet MSVC
            private:

                Mat &markerObjPoints;
                InputArrayOfArrays corners;
                InputArray cameraMatrix, distCoeffs;
                Mat &rvecs, tvecs;
            };


            void estimatePoseSingleMarkers(InputArrayOfArrays _corners, float markerLength,
                                           InputArray _cameraMatrix, InputArray _distCoeffs,
                                           OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs) {

                CV_Assert(markerLength > 0);

                Mat markerObjPoints;
                _getSingleMarkerObjectPoints(markerLength, markerObjPoints);
                int nMarkers = (int) _corners.total();
                _rvecs.create(nMarkers, 1, CV_64FC3);
                _tvecs.create(nMarkers, 1, CV_64FC3);

                Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();

                //// for each marker, calculate its pose
                // for (int i = 0; i < nMarkers; i++) {
                //    solvePnP(markerObjPoints, _corners.getMat(i), _cameraMatrix, _distCoeffs,
                //             _rvecs.getMat(i), _tvecs.getMat(i));
                //}

                // this is the parallel call for the previous commented loop (result is equivalent)
                parallel_for_(Range(0, nMarkers),
                              SinglePoseEstimationParallel(markerObjPoints, _corners, _cameraMatrix,
                                                           _distCoeffs, rvecs, tvecs));
            }


            /**
            * @brief Given a board configuration and a set of detected markers, returns the corresponding
            * image points and object points to call solvePnP
            */
            static void _getBoardObjectAndImagePoints(const Board &board, InputArray _detectedIds,
                                                      InputArrayOfArrays _detectedCorners,
                                                      OutputArray _imgPoints, OutputArray _objPoints) {

                CV_Assert(board.ids.size() == board.objPoints.size());
                CV_Assert(_detectedIds.total() == _detectedCorners.total());

                int nDetectedMarkers = (int) _detectedIds.total();

                vector<Point3f> objPnts;
                objPnts.reserve(nDetectedMarkers);

                vector<Point2f> imgPnts;
                imgPnts.reserve(nDetectedMarkers);

                auto itemAmount = board.dictionary.bytesList.rows;

                // look for detected markers that belong to the board and get their information
                for (int i = 0; i < nDetectedMarkers; i++) {
                    int currentId = _detectedIds.getMat().ptr<int>(0)[i];
                    bool need_flip = currentId / itemAmount;//是否需要翻转
                    currentId %= itemAmount;
                    for (unsigned int j = 0; j < board.ids.size(); j++) {
                        if (currentId == board.ids[j]) {
                            if (need_flip)
                                for (int p = 0; p < 4; p++) {
                                    const auto &point = board.objPoints[j][p];
                                    /// 三维笛卡尔坐标系的平面反向
                                    objPnts.emplace_back(-point.x, point.y, -point.z);
                                    imgPnts.push_back(_detectedCorners.getMat(i).ptr<Point2f>(0)[p]);
                                }
                            else
                                for (int p = 0; p < 4; p++) {
                                    objPnts.push_back(board.objPoints[j][p]);
                                    imgPnts.push_back(_detectedCorners.getMat(i).ptr<Point2f>(0)[p]);
                                }
                        }
                    }
                }

                // create output
                _objPoints.create((int) objPnts.size(), 1, CV_32FC3);
                for (unsigned int i = 0; i < objPnts.size(); i++)
                    _objPoints.getMat().ptr<Point3f>(0)[i] = objPnts[i];

                _imgPoints.create((int) objPnts.size(), 1, CV_32FC2);
                for (unsigned int i = 0; i < imgPnts.size(); i++)
                    _imgPoints.getMat().ptr<Point2f>(0)[i] = imgPnts[i];
            }


            int estimatePoseBoard(InputArrayOfArrays _corners, InputArray _ids, const Board &board,
                                  InputArray _cameraMatrix, InputArray _distCoeffs, OutputArray _rvec,
                                  OutputArray _tvec) {

                CV_Assert(_corners.total() == _ids.total());

                // get object and image points for the solvePnP function
                Mat objPoints, imgPoints;
                _getBoardObjectAndImagePoints(board, _ids, _corners, imgPoints, objPoints);

                CV_Assert(imgPoints.total() == objPoints.total());

                if (objPoints.total() == 0) // 0 of the detected markers in board
                    return 0;

                _rvec.create(3, 1, CV_64FC1);
                _tvec.create(3, 1, CV_64FC1);
                solvePnP(objPoints, imgPoints, _cameraMatrix, _distCoeffs, _rvec, _tvec);

                // divide by four since all the four corners are concatenated in the array for each marker
                return (int) objPoints.total() / 4;
            }


            void drawDetectedMarkers(InputOutputArray _image, InputArrayOfArrays _corners,
                                     InputArray _ids, const Scalar &borderColor) {


                CV_Assert(_image.getMat().total() != 0 &&
                          (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
                CV_Assert((_corners.total() == _ids.total()) || _ids.total() == 0);

                // calculate colors
                Scalar textColor, cornerColor;
                textColor = cornerColor = borderColor;
                swap(textColor.val[0], textColor.val[1]);     // text color just sawp G and R
                swap(cornerColor.val[1], cornerColor.val[2]); // corner color just sawp G and B

                int nMarkers = (int) _corners.total();
                for (int i = 0; i < nMarkers; i++) {
                    Mat currentMarker = _corners.getMat(i);
                    CV_Assert(currentMarker.total() == 4 && currentMarker.type() == CV_32FC2);

                    // draw marker sides
                    for (int j = 0; j < 4; j++) {
                        Point2f p0, p1;
                        p0 = currentMarker.ptr<Point2f>(0)[j];
                        p1 = currentMarker.ptr<Point2f>(0)[(j + 1) % 4];
                        line(_image, p0, p1, borderColor, 1);
                    }
                    // draw first corner mark
                    rectangle(_image, currentMarker.ptr<Point2f>(0)[0] - Point2f(3, 3),
                              currentMarker.ptr<Point2f>(0)[0] + Point2f(3, 3), cornerColor, 1, LINE_AA);

                    // draw ID
                    if (_ids.total() != 0) {
                        Point2f cent(0, 0);
                        for (int p = 0; p < 4; p++)
                            cent += currentMarker.ptr<Point2f>(0)[p];
                        cent = cent / 4.;
                        stringstream s;
                        s << "id=" << _ids.getMat().ptr<int>(0)[i];
                        putText(_image, s.str(), cent, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);
                    }
                }
            }


            void drawAxis(InputOutputArray _image, InputArray _cameraMatrix, InputArray _distCoeffs,
                          InputArray _rvec, InputArray _tvec, float length) {

                CV_Assert(_image.getMat().total() != 0 &&
                          (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
                CV_Assert(length > 0);

                // project axis points
                vector<Point3f> axisPoints;
                axisPoints.emplace_back(0, 0, 0);
                axisPoints.emplace_back(length, 0, 0);
                axisPoints.emplace_back(0, length, 0);
                axisPoints.emplace_back(0, 0, length);
                vector<Point2f> imagePoints;
                projectPoints(axisPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

                // draw axis lines
                line(_image, imagePoints[0], imagePoints[1], Scalar(0, 0, 255), 3);
                line(_image, imagePoints[0], imagePoints[2], Scalar(0, 255, 0), 3);
                line(_image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0), 3);
            }


            void drawMarker(const Dictionary &dictionary, int id, int sidePixels, OutputArray _img, int borderBits) {
                dictionary.drawMarker(id, sidePixels, _img, borderBits);
            }

        }
    }
}

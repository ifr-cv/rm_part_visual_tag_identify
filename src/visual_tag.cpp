//
// Created by yuanlu on 2023/2/1.
//
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/imgproc.hpp>
#include "visual_tag.hpp"

namespace cv {
    namespace visual_tag {


        DetectorParameters::DetectorParameters()
                : minMarkerPerimeterRate(0.03),
                  maxMarkerPerimeterRate(4.),
                  polygonalApproxAccuracyRate(0.03),
                  minCornerDistanceRate(0.05),
                  minDistanceToBorder(3),
                  minMarkerDistanceRate(0.05),
                  doCornerRefinement(false),
                  cornerRefinementWinSize(5),
                  cornerRefinementMaxIterations(30),
                  cornerRefinementMinAccuracy(0.1),
                  markerBorderBits(1),
                  perspectiveRemovePixelPerCell(4),
                  perspectiveRemoveIgnoredMarginPerCell(0.13),
                  maxErroneousBitsInBorderRate(0.35),
                  minOtsuStdDev(5.0),
                  errorCorrectionRate(0.6) {}

        Dictionary::Dictionary(const cv::Mat &_bytesList, int _markerSize, int _maxcorr) {
            markerSize = _markerSize;
            maxCorrectionBits = _maxcorr;
            bytesList = _bytesList;
        }


        bool Dictionary::identify(const cv::Mat &onlyBits, int &idx, int &rotation,
                                  double maxCorrectionRate) const {

            CV_Assert(onlyBits.rows == markerSize && onlyBits.cols == markerSize);

            int maxCorrectionRecalculed = int(double(maxCorrectionBits) * maxCorrectionRate);

            // get as a byte list
            cv::Mat candidateBytes = getByteListFromBits(onlyBits);

            idx = -1; // by default, not found

            // search closest marker in dict
            for (int m = 0; m < bytesList.rows; m++) {
                int currentMinDistance = markerSize * markerSize + 1;
                int currentRotation = -1;
                for (unsigned int r = 0; r < 4; r++) {
                    int currentHamming = cv::hal::normHamming(
                            bytesList.ptr(m) + r * candidateBytes.cols,
                            candidateBytes.ptr(),
                            candidateBytes.cols);

                    if (currentHamming < currentMinDistance) {
                        currentMinDistance = currentHamming;
                        currentRotation = r;
                    }
                }

                // if maxCorrection is fullfilled, return this one
                if (currentMinDistance <= maxCorrectionRecalculed) {
                    idx = m;
                    rotation = currentRotation;
                    break;
                }
            }

            return idx != -1;
        }


        int Dictionary::getDistanceToId(InputArray bits, int id, bool allRotations) const {

            CV_Assert(id >= 0 && id < bytesList.rows);

            unsigned int nRotations = 4;
            if (!allRotations) nRotations = 1;

            cv::Mat candidateBytes = getByteListFromBits(bits.getMat());
            int currentMinDistance = int(bits.total() * bits.total());
            for (unsigned int r = 0; r < nRotations; r++) {
                int currentHamming = cv::hal::normHamming(
                        bytesList.ptr(id) + r * candidateBytes.cols,
                        candidateBytes.ptr(),
                        candidateBytes.cols);

                if (currentHamming < currentMinDistance) {
                    currentMinDistance = currentHamming;
                }
            }
            return currentMinDistance;
        }


        void Dictionary::drawMarker(int id, int sidePixels, OutputArray _img, int borderBits) const {

            CV_Assert(sidePixels > markerSize);
            CV_Assert(id < bytesList.rows);
            CV_Assert(borderBits > 0);

            _img.create(sidePixels, sidePixels, CV_8UC1);

            // create small marker with 1 pixel per bin
            cv::Mat tinyMarker(markerSize + 2 * borderBits, markerSize + 2 * borderBits, CV_8UC1,
                               Scalar::all(0));
            cv::Mat innerRegion = tinyMarker.rowRange(borderBits, tinyMarker.rows - borderBits)
                    .colRange(borderBits, tinyMarker.cols - borderBits);
            // put inner bits
            cv::Mat bits = 255 * getBitsFromByteList(bytesList.rowRange(id, id + 1), markerSize);
            CV_Assert(innerRegion.total() == bits.total());
            bits.copyTo(innerRegion);

            // resize tiny marker to output size
            cv::resize(tinyMarker, _img.getMat(), _img.getMat().size(), 0, 0, INTER_NEAREST);
        }


        cv::Mat Dictionary::getByteListFromBits(const cv::Mat &bits) {
            // integer ceil
            int nbytes = (bits.cols * bits.rows + 8 - 1) / 8;

            cv::Mat candidateByteList(1, nbytes, CV_8UC4, Scalar::all(0));
            unsigned char currentBit = 0;
            int currentByte = 0;

            // the 4 rotations
            uchar *rot0 = candidateByteList.ptr();
            uchar *rot1 = candidateByteList.ptr() + 1 * nbytes;
            uchar *rot2 = candidateByteList.ptr() + 2 * nbytes;
            uchar *rot3 = candidateByteList.ptr() + 3 * nbytes;

            for (int row = 0; row < bits.rows; row++) {
                for (int col = 0; col < bits.cols; col++) {
                    // circular shift
                    rot0[currentByte] <<= 1;
                    rot1[currentByte] <<= 1;
                    rot2[currentByte] <<= 1;
                    rot3[currentByte] <<= 1;
                    // set bit
                    rot0[currentByte] |= bits.at<uchar>(row, col);
                    rot1[currentByte] |= bits.at<uchar>(col, bits.cols - 1 - row);
                    rot2[currentByte] |= bits.at<uchar>(bits.rows - 1 - row, bits.cols - 1 - col);
                    rot3[currentByte] |= bits.at<uchar>(bits.rows - 1 - col, row);
                    currentBit++;
                    if (currentBit == 8) {
                        // next byte
                        currentBit = 0;
                        currentByte++;
                    }
                }
            }
            return candidateByteList;
        }


        cv::Mat Dictionary::getBitsFromByteList(const cv::Mat &byteList, int markerSize) {
            CV_Assert(byteList.total() > 0 &&
                      byteList.total() >= (unsigned int) markerSize * markerSize / 8 &&
                      byteList.total() <= (unsigned int) markerSize * markerSize / 8 + 1);
            cv::Mat bits(markerSize, markerSize, CV_8UC1, Scalar::all(0));

            unsigned char base2List[] = {128, 64, 32, 16, 8, 4, 2, 1};
            int currentByteIdx = 0;
            // we only need the bytes in normal rotation
            unsigned char currentByte = byteList.ptr()[0];
            int currentBit = 0;
            for (int row = 0; row < bits.rows; row++) {
                for (int col = 0; col < bits.cols; col++) {
                    if (currentByte >= base2List[currentBit]) {
                        bits.at<unsigned char>(row, col) = 1;
                        currentByte -= base2List[currentBit];
                    }
                    currentBit++;
                    if (currentBit == 8) {
                        currentByteIdx++;
                        currentByte = byteList.ptr()[currentByteIdx];
                        // if not enough bits for one more byte, we are in the end
                        // update bit position accordingly
                        if (8 * (currentByteIdx + 1) > (int) bits.total())
                            currentBit = 8 * (currentByteIdx + 1) - (int) bits.total();
                        else
                            currentBit = 0; // ok, bits enough for next byte
                    }
                }
            }
            return bits;
        }


        /**
        * @brief Generates a random marker GpuMat of size markerSize x markerSize
        */
        static cv::Mat _generateRandomMarker(int markerSize) {
            cv::Mat marker(markerSize, markerSize, CV_8UC1, Scalar::all(0));
            for (int i = 0; i < markerSize; i++) {
                for (int j = 0; j < markerSize; j++) {
                    unsigned char bit = rand() % 2;
                    marker.at<unsigned char>(i, j) = bit;
                }
            }
            return marker;
        }

        /**
        * @brief Calculate selfDistance of the codification of a marker GpuMat. Self distance is the Hamming
        * distance of the marker to itself in the other rotations.
        * See S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez. 2014.
        * "Automatic generation and detection of highly reliable fiducial markers under occlusion".
        * Pattern Recogn. 47, 6 (June 2014), 2280-2292. DOI=10.1016/j.patcog.2014.01.005
        */
        static int _getSelfDistance(const cv::Mat &marker) {
            cv::Mat bytes = Dictionary::getByteListFromBits(marker);
            int minHamming = (int) marker.total() + 1;
            for (int r = 1; r < 4; r++) {
                int currentHamming = cv::hal::normHamming(bytes.ptr(), bytes.ptr() + bytes.cols * r, bytes.cols);
                if (currentHamming < minHamming) minHamming = currentHamming;
            }
            return minHamming;
        }

        Dictionary generateCustomDictionary(int nMarkers, int markerSize,
                                            const Dictionary &baseDictionary) {

            Dictionary out;
            out.markerSize = markerSize;

            // theoretical maximum intermarker distance
            // See S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez. 2014.
            // "Automatic generation and detection of highly reliable fiducial markers under occlusion".
            // Pattern Recogn. 47, 6 (June 2014), 2280-2292. DOI=10.1016/j.patcog.2014.01.005
            int C = (int) std::floor(float(markerSize * markerSize) / 4.f);
            int tau = 2 * (int) std::floor(float(C) * 4.f / 3.f);

            // if baseDictionary is provided, calculate its intermarker distance
            if (baseDictionary.bytesList.rows > 0) {
                CV_Assert(baseDictionary.markerSize == markerSize);
                out.bytesList = baseDictionary.bytesList.clone();

                int minDistance = markerSize * markerSize + 1;
                for (int i = 0; i < out.bytesList.rows; i++) {
                    cv::Mat markerBytes = out.bytesList.rowRange(i, i + 1);
                    cv::Mat markerBits = Dictionary::getBitsFromByteList(markerBytes, markerSize);
                    minDistance = std::min(minDistance, _getSelfDistance(markerBits));
                    for (int j = i + 1; j < out.bytesList.rows; j++) {
                        minDistance = std::min(minDistance, out.getDistanceToId(markerBits, j));
                    }
                }
                tau = minDistance;
            }

            // current best option
            int bestTau = 0;
            cv::Mat bestMarker;

            // after these number of unproductive iterations, the best option is accepted
            const int maxUnproductiveIterations = 5000;
            int unproductiveIterations = 0;

            while (out.bytesList.rows < nMarkers) {
                cv::Mat currentMarker = _generateRandomMarker(markerSize);

                int selfDistance = _getSelfDistance(currentMarker);
                int minDistance = selfDistance;

                // if self distance is better or equal than current best option, calculate distance
                // to previous accepted markers
                if (selfDistance >= bestTau) {
                    for (int i = 0; i < out.bytesList.rows; i++) {
                        int currentDistance = out.getDistanceToId(currentMarker, i);
                        minDistance = std::min(currentDistance, minDistance);
                        if (minDistance <= bestTau) {
                            break;
                        }
                    }
                }

                // if distance is high enough, accept the marker
                if (minDistance >= tau) {
                    unproductiveIterations = 0;
                    bestTau = 0;
                    cv::Mat bytes = Dictionary::getByteListFromBits(currentMarker);
                    out.bytesList.push_back(bytes);
                } else {
                    unproductiveIterations++;

                    // if distance is not enough, but is better than the current best option
                    if (minDistance > bestTau) {
                        bestTau = minDistance;
                        bestMarker = currentMarker;
                    }

                    // if number of unproductive iterarions has been reached, accept the current best option
                    if (unproductiveIterations == maxUnproductiveIterations) {
                        unproductiveIterations = 0;
                        tau = bestTau;
                        bestTau = 0;
                        cv::Mat bytes = Dictionary::getByteListFromBits(bestMarker);
                        out.bytesList.push_back(bytes);
                    }
                }
            }

            // update the maximum number of correction bits for the generated dictionary
            out.maxCorrectionBits = (tau - 1) / 2;

            return out;
        }

        Dictionary loadDictionaryFromFile(const cv::String &filename) {
            cv::FileStorage fs(filename, cv::FileStorage::Mode::READ);
            if (!fs.isOpened())return Dictionary();

            Dictionary dict(fs["ByteList"].mat(), fs["MarkerSize"]);
            auto MaxCorrectionBits = fs["MaxCorrectionBits"];
            if (MaxCorrectionBits.isInt())dict.maxCorrectionBits = MaxCorrectionBits;
            return dict;
        }

        void generateByteFile(const cv::String &bitYaml, const cv::String &byteYaml, const std::string &prefix, std::size_t amount) {
            CV_Assert(amount > 0);
            cv::FileStorage fs(bitYaml, cv::FileStorage::Mode::READ);

            cv::Size size;
            for (std::size_t i = 0; i < amount; i++) {
                auto mat = fs[prefix + std::to_string(i)].mat();
                if (size.empty())
                    size = mat.size();
                else
                    CV_Assert(size == mat.size());
            }
            CV_Assert(size.width == size.height);

            int nbytes = (size.width * size.height + 8 - 1) / 8;
            cv::Mat byteList(amount, nbytes, CV_8UC4);

            for (std::size_t i = 0; i < amount; i++) {
                auto mat = fs[prefix + std::to_string(i)].mat();
                cv::Mat mat_8u;
                mat.convertTo(mat_8u, CV_8UC1);
                auto bl = Dictionary::getByteListFromBits(mat_8u);
                bl.copyTo(byteList.row(i));
            }

            saveDictionaryToFile(byteYaml, Dictionary(byteList, size.width));
        }

        void saveDictionaryToFile(const String &filename, const Dictionary &dictionary) {
            cv::FileStorage fs_out(filename, cv::FileStorage::Mode::WRITE);
            fs_out << "ByteList" << dictionary.bytesList;
            fs_out << "MarkerSize" << dictionary.markerSize;
            fs_out << "MaxCorrectionBits" << dictionary.maxCorrectionBits;
        }

    }
}
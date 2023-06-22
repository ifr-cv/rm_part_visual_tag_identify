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

#ifndef __OPENCV_VISUAL_TAG_CPU_HANDLER_HPP__
#define __OPENCV_VISUAL_TAG_CPU_HANDLER_HPP__

#include <opencv2/core.hpp>
#include <vector>
#include "visual_tag.hpp"

/**
* @defgroup aruco ArUco Marker Detection
* This module is dedicated to square fiducial markers (also known as Augmented Reality Markers)
* These markers are useful for easy, fast and robust camera pose estimation.ç
*
* The main functionalities are:
* - Detection of markers in a image
* - Pose estimation from a single marker or from a board/set of markers
* - Detection of ChArUco board for high subpixel accuracy
* - Camera calibration from both, ArUco boards and ChArUco boards.
* - Detection of ChArUco diamond markers
* The samples directory includes easy examples of how to use the module.
*
* The implementation is based on the ArUco Library by R. Muñoz-Salinas and S. Garrido-Jurado.
*
* @sa S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez. 2014.
* "Automatic generation and detection of highly reliable fiducial markers under occlusion".
* Pattern Recogn. 47, 6 (June 2014), 2280-2292. DOI=10.1016/j.patcog.2014.01.005
*
* @sa http://www.uco.es/investiga/grupos/ava/node/26
*
* This module has been originally developed by Sergio Garrido-Jurado as a project
* for Google Summer of Code 2015 (GSoC 15).
*
*
*/

namespace cv {
    namespace cpu {
        namespace visual_tag {
            using namespace cv::visual_tag;

            /**
            * @brief Basic marker detection
            *
            * @param image input image
            * @param dictionary indicates the type of markers that will be searched
            * @param corners vector of detected marker corners. For each marker, its four corners
            * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
            * the dimensions of this array is Nx4. The order of the corners is clockwise.
            * @param ids vector of identifiers of the detected markers. The identifier is of type int
            * (e.g. std::vector<int>). For N detected markers, the size of ids is also N.
            * The identifiers have the same order than the markers in the imgPoints array.
            * @param parameters marker detection parameters
            * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a
            * correct codification. Useful for debugging purposes.
            *
            * Performs marker detection in the input image. Only markers included in the specific dictionary
            * are searched. For each detected marker, it returns the 2D position of its corner in the image
            * and its corresponding identifier.
            * Note that this function does not perform pose estimation.
            * @sa estimatePoseSingleMarkers,  estimatePoseBoard
            *
            */
            CV_EXPORTS void detectMarkers(InputArray image, const Dictionary &dictionary, OutputArrayOfArrays corners,
                                          OutputArray ids, DetectorParameters parameters = DetectorParameters(),
                                          OutputArrayOfArrays rejectedImgPoints = noArray());



            /**
            * @brief Pose estimation for single markers
            *
            * @param corners vector of already detected markers corners. For each marker, its four corners
            * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
            * the dimensions of this array should be Nx4. The order of the corners should be clockwise.
            * @sa detectMarkers
            * @param markerLength the length of the markers' side. The returning translation vectors will
            * be in the same unit. Normally, unit is meters.
            * @param cameraMatrix input 3x3 floating-point camera matrix
            * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
            * @param distCoeffs vector of distortion coefficients
            * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
            * @param rvecs array of output rotation vectors (@sa Rodrigues) (e.g. std::vector<cv::Vec3d>>).
            * Each element in rvecs corresponds to the specific marker in imgPoints.
            * @param tvecs array of output translation vectors (e.g. std::vector<cv::Vec3d>>).
            * Each element in tvecs corresponds to the specific marker in imgPoints.
            *
            * This function receives the detected markers and returns their pose estimation respect to
            * the camera individually. So for each marker, one rotation and translation vector is returned.
            * The returned transformation is the one that transforms points from each marker coordinate system
            * to the camera coordinate system.
            * The marker corrdinate system is centered on the middle of the marker, with the Z axis
            * perpendicular to the marker plane.
            * The coordinates of the four corners of the marker in its own coordinate system are:
            * (-markerLength/2, markerLength/2, 0), (markerLength/2, markerLength/2, 0),
            * (markerLength/2, -markerLength/2, 0), (-markerLength/2, -markerLength/2, 0)
            */
            CV_EXPORTS void estimatePoseSingleMarkers(InputArrayOfArrays corners, float markerLength,
                                                      InputArray cameraMatrix, InputArray distCoeffs,
                                                      OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs);

            /**
            * @brief Pose estimation for a board of markers
            *
            * @param corners vector of already detected markers corners. For each marker, its four corners
            * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the
            * dimensions of this array should be Nx4. The order of the corners should be clockwise.
            * @param ids list of identifiers for each marker in corners
            * @param board layout of markers in the board. The layout is composed by the marker identifiers
            * and the positions of each marker corner in the board reference system.
            * @param cameraMatrix input 3x3 floating-point camera matrix
            * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
            * @param distCoeffs vector of distortion coefficients
            * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
            * @param rvec Output vector (e.g. cv::GpuMat) corresponding to the rotation vector of the board
            * (@sa Rodrigues).
            * @param tvec Output vector (e.g. cv::GpuMat) corresponding to the translation vector of the board.
            *
            * This function receives the detected markers and returns the pose of a marker board composed
            * by those markers.
            * A Board of marker has a single world coordinate system which is defined by the board layout.
            * The returned transformation is the one that transforms points from the board coordinate system
            * to the camera coordinate system.
            * Input markers that are not included in the board layout are ignored.
            * The function returns the number of markers from the input employed for the board pose estimation.
            * Note that returning a 0 means the pose has not been estimated.
            */
            CV_EXPORTS int estimatePoseBoard(InputArrayOfArrays corners, InputArray ids, const Board &board,
                                             InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec,
                                             OutputArray tvec);

            /**
            * @brief Draw detected markers in image
            *
            * @param image input/output image. It must have 1 or 3 channels. The number of channels is not
            * altered.
            * @param corners positions of marker corners on input image.
            * (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of
            * this array should be Nx4. The order of the corners should be clockwise.
            * @param ids vector of identifiers for markers in markersCorners .
            * Optional, if not provided, ids are not painted.
            * @param borderColor color of marker borders. Rest of colors (text color and first corner color)
            * are calculated based on this one to improve visualization.
            *
            * Given an array of detected marker corners and its corresponding ids, this functions draws
            * the markers in the image. The marker borders are painted and the markers identifiers if provided.
            * Useful for debugging purposes.
            */
            CV_EXPORTS void drawDetectedMarkers(InputOutputArray image, InputArrayOfArrays corners,
                                                InputArray ids = noArray(),
                                                const Scalar &borderColor = Scalar(0, 255, 0));



            /**
            * @brief Draw coordinate system axis from pose estimation
            *
            * @param image input/output image. It must have 1 or 3 channels. The number of channels is not
            * altered.
            * @param cameraMatrix input 3x3 floating-point camera matrix
            * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
            * @param distCoeffs vector of distortion coefficients
            * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
            * @param rvec rotation vector of the coordinate system that will be drawn. (@sa Rodrigues).
            * @param tvec translation vector of the coordinate system that will be drawn.
            * @param length length of the painted axis in the same unit than tvec (usually in meters)
            *
            * Given the pose estimation of a marker or board, this function draws the axis of the world
            * coordinate system, i.e. the system centered on the marker/board. Useful for debugging purposes.
            */
            CV_EXPORTS void drawAxis(InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                                     InputArray rvec, InputArray tvec, float length);



            /**
            * @brief Draw a canonical marker image
            *
            * @param dictionary dictionary of markers indicating the type of markers
            * @param id identifier of the marker that will be returned. It has to be a valid id
            * in the specified dictionary.
            * @param sidePixels size of the image in pixels
            * @param img output image with the marker
            * @param borderBits width of the marker border.
            *
            * This function returns a marker image in its canonical form (i.e. ready to be printed)
            */
            CV_EXPORTS void drawMarker(const Dictionary &dictionary, int id, int sidePixels, OutputArray img,
                                       int borderBits = 1);
        }
    }
}

#endif //__OPENCV_VISUAL_TAG_CPU_HANDLER_HPP__

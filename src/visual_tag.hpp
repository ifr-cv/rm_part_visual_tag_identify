//
// Created by yuanlu on 2023/2/1.
//

#ifndef __OPENCV_VISUAL_TAG_DEF_HANDLER_HPP__
#define __OPENCV_VISUAL_TAG_DEF_HANDLER_HPP__

#include <opencv2/core.hpp>

namespace cv {
    namespace visual_tag {

        /**
        * @brief Dictionary/Set of markers. It contains the inner codification
        *
        * bytesList contains the marker codewords where
        * - bytesList.rows is the dictionary size
        * - each marker is encoded using `nbytes = ceil(markerSize*markerSize/8.)`
        * - each row contains all 4 rotations of the marker, so its length is `4*nbytes`
        *
        * `bytesList.ptr(i)[k*nbytes + j]` is then the j-th byte of i-th marker, in its k-th rotation.
        */
        class CV_EXPORTS Dictionary {

        public:
            cv::Mat bytesList;         // marker code information
            int markerSize;        // number of bits per dimension
            int maxCorrectionBits; // maximum number of bits that can be corrected

            explicit Dictionary(const cv::Mat &_bytesList = cv::Mat(), int _markerSize = 0, int _maxcorr = 0);

            /**
            * @brief Given a matrix of bits. Returns whether if marker is identified or not.
            * It returns by reference the correct id (if any) and the correct rotation
            */
            bool identify(const cv::Mat &onlyBits, int &idx, int &rotation, double maxCorrectionRate) const;

            /**
            * @brief Returns the distance of the input bits to the specific id. If allRotations is true,
            * the four posible bits rotation are considered
            */
            int getDistanceToId(InputArray bits, int id, bool allRotations = true) const;


            /**
            * @brief Draw a canonical marker image
            */
            void drawMarker(int id, int sidePixels, OutputArray _img, int borderBits = 1) const;


            /**
            * @brief Transform matrix of bits to list of bytes in the 4 rotations
            */
            static cv::Mat getByteListFromBits(const cv::Mat &bits);


            /**
            * @brief Transform list of bytes to matrix of bits
            */
            static cv::Mat getBitsFromByteList(const cv::Mat &byteList, int markerSize);
        };

        /**
        * @brief Board of markers
        *
        * A board is a set of markers in the 3D space with a common cordinate system.
        * The common form of a board of marker is a planar (2D) board, however any 3D layout can be used.
        * A Board object is composed by:
        * - The object points of the marker corners, i.e. their coordinates respect to the board system.
        * - The dictionary which indicates the type of markers of the board
        * - The identifier of all the markers in the board.
        */
        class CV_EXPORTS Board {

        public:
            // array of object points of all the marker corners in the board
            // each marker include its 4 corners, i.e. for M markers, the size is Mx4
            std::vector<std::vector<Point3f> > objPoints;

            // the dictionary of markers employed for this board
            Dictionary dictionary;

            // vector of the identifiers of the markers in the board (same size than objPoints)
            // The identifiers refers to the board dictionary
            std::vector<int> ids;
        };


        /**
         * @brief 从文件中读取标记字典
         * @param filename 文件路径
         * @return 读取的字典
         *
         * 文件应由cv::FileStorage写出, 且应包含ByteList (U4类型的Mat) 及 MarkerSize (int)
         */
        Dictionary loadDictionaryFromFile(const cv::String &filename);

        /**
         * @brief 保存标记字典到文件
         * @param filename 文件路径
         * @param dictionary 待保存的字典
         */
        void saveDictionaryToFile(const cv::String &filename, const Dictionary &dictionary);

        /**
         * @brief 由矩阵文件生成可由 loadDictionaryFromFile 读取的编码文件
         * @param bitYaml 矩阵文件
         * @param byteYaml 编码文件
         * @param prefix 矩阵文件中的节点前缀
         * @param amount 矩阵数量
         */
        void generateByteFile(const cv::String &bitYaml, const cv::String &byteYaml, const std::string &prefix, std::size_t amount);

        /**
        * @brief Generates a new customizable marker dictionary
        *
        * @param nMarkers number of markers in the dictionary
        * @param markerSize number of bits per dimension of each markers
        * @param baseDictionary Include the markers in this dictionary at the beginning (optional)
        *
        * This function creates a new dictionary composed by nMarkers markers and each markers composed
        * by markerSize x markerSize bits. If baseDictionary is provided, its markers are directly
        * included and the rest are generated based on them. If the size of baseDictionary is higher
        * than nMarkers, only the first nMarkers in baseDictionary are taken and no new marker is added.
        */
        CV_EXPORTS Dictionary

        generateCustomDictionary(int nMarkers, int markerSize, const Dictionary &baseDictionary = Dictionary());

        ///Parameters for the detectMarker process
        struct CV_EXPORTS DetectorParameters {

            DetectorParameters();

            double minMarkerPerimeterRate;//determine minimum perimeter for marker contour to be detected. This is defined as a rate respect to the maximum dimension of the input image (default 0.03).
            double maxMarkerPerimeterRate;//determine maximum perimeter for marker contour to be detected. This is defined as a rate respect to the maximum dimension of the input image (default 4.0).
            double polygonalApproxAccuracyRate;//minimum accuracy during the polygonal approximation process to determine which contours are squares.
            double minCornerDistanceRate;//minimum distance between corners for detected markers relative to its perimeter (default 0.05)
            int minDistanceToBorder;//minimum distance of any corner to the image border for detected markers (in pixels) (default 3)
            double minMarkerDistanceRate;//minimum mean distance beetween two marker corners to be considered similar, so that the smaller one is removed. The rate is relative to the smaller perimeter of the two markers (default 0.05).
            bool doCornerRefinement;//do subpixel refinement or not
            int cornerRefinementWinSize;//window size for the corner refinement process (in pixels) (default 5).
            int cornerRefinementMaxIterations;
            double cornerRefinementMinAccuracy;//maximum number of iterations for stop criteria of the corner refinement process (default 30).
            int markerBorderBits;//minimum error for the stop cristeria of the corner refinement process (default: 0.1)
            int perspectiveRemovePixelPerCell;//number of bits of the marker border, i.e. marker border width (default 1).
            double perspectiveRemoveIgnoredMarginPerCell;//number of bits (per dimension) for each cell of the marker when removing the perspective (default 8).
            double maxErroneousBitsInBorderRate;//width of the margin of pixels on each cell not considered for the determination of the cell bit. Represents the rate respect to the total size of the cell, i.e. perpectiveRemovePixelPerCell (default 0.13)
            double minOtsuStdDev;//maximum number of accepted erroneous bits in the border (i.e. number of allowed white bits in the border). Represented as a rate respect to the total number of bits per marker (default 0.35).
            double errorCorrectionRate;//minimun standard deviation in pixels values during the decodification step to apply Otsu thresholding (otherwise, all the bits are set to 0 or 1 depending on mean higher than 128 or not) (default 5.0)
        };


    }
}
#endif //__OPENCV_VISUAL_TAG_DEF_HANDLER_HPP__

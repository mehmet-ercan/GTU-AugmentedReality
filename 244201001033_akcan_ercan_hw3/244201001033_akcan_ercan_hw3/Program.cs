using System.Drawing; // Requires System.Drawing.Common package for image processing
using System.Runtime.Versioning;
using MathNet.Numerics.LinearAlgebra.Double;

namespace _244201001033_akcan_ercan_hw3
{
    class Program
    {
        [SupportedOSPlatform("windows")]
        static void Main(string[] args)
        {
            // Load the images
            string inputImagePath = "Homework_3_img3.JPG";
            string checkerImagePath = "checkerboard_7x9_700x900.jpg";

            Bitmap inputImage = new(inputImagePath);
            Bitmap checkerImage = new(checkerImagePath);

            // var srcPoints_im1 = new double[,] {
            //     { 476, 518 },
            //     { 2088, 540 },
            //     { 2070, 2620 },
            //     { 450, 2610 }
            // };

            // var srcPoints_im2 = new double[,] {
            //     { 285, 510 },
            //     { 1948, 448 },
            //     { 1925, 2695 },
            //     { 250, 2608 }
            // };

            var srcPoints_im3 = new double[,] {
                { 493, 596 },
                { 2145, 721 },
                { 1945, 2619 },
                { 489, 2677 }
            };

            var dstPoints = new double[,] {
                { 0, 0 },
                { 700, 0 },
                { 700, 900 },
                { 0, 900 }
            };

            double[,] homographyMatrix = FindHomography(srcPoints_im3, dstPoints);
            // double[,]? homographyMatrixRansac = FindHomographyRansac(srcPoints_im2, dstPoints);
            Bitmap outputImage = WarpImage(inputImage, homographyMatrix, checkerImage.Width, checkerImage.Height);
            // Bitmap outputImageRansac = WarpImage(inputImage, homographyMatrixRansac, checkerImage.Width, checkerImage.Height);

            outputImage.Save("warpedImage.jpg");
            // outputImageRansac.Save("warpedImageRansac.jpg");
            Console.WriteLine($"Warped image saved at: warpedImage");
        }

        [SupportedOSPlatform("windows")]
        public static double[,] FindHomography(double[,] srcPoints, double[,] dstPoints)
        {
            if (srcPoints.GetLength(0) != 4 || dstPoints.GetLength(0) != 4)
            {
                throw new ArgumentException("Exactly 4 points are required for both source and destination.");
            }

            // Create the matrix A (8x9)
            double[,] A = new double[8, 9];
            for (int i = 0; i < 4; i++)
            {
                double x = srcPoints[i, 0];
                double y = srcPoints[i, 1];
                double xPrime = dstPoints[i, 0];
                double yPrime = dstPoints[i, 1];

                // First row for this point pair
                A[2 * i, 0] = -x;
                A[2 * i, 1] = -y;
                A[2 * i, 2] = -1;
                A[2 * i, 3] = 0;
                A[2 * i, 4] = 0;
                A[2 * i, 5] = 0;
                A[2 * i, 6] = x * xPrime;
                A[2 * i, 7] = y * xPrime;
                A[2 * i, 8] = xPrime;

                // Second row for this point pair
                A[2 * i + 1, 0] = 0;
                A[2 * i + 1, 1] = 0;
                A[2 * i + 1, 2] = 0;
                A[2 * i + 1, 3] = -x;
                A[2 * i + 1, 4] = -y;
                A[2 * i + 1, 5] = -1;
                A[2 * i + 1, 6] = x * yPrime;
                A[2 * i + 1, 7] = y * yPrime;
                A[2 * i + 1, 8] = yPrime;
            }

            var matrixA = DenseMatrix.OfArray(A);
            var svd = matrixA.Svd();
            var v = svd.VT.Transpose();
            var h = v.Column(v.ColumnCount - 1).ToArray();

            // h into a 3x3 matrix
            double[,] H = new double[3, 3];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    H[i, j] = h[i * 3 + j];
                }
            }

            double scale = H[2, 2];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    H[i, j] /= scale;
                }
            }

            return H;
        }

        [SupportedOSPlatform("windows")]
        public static double[,]? FindHomographyRansac(double[,] srcPoints, double[,] dstPoints, int maxIterations = 100000, double threshold = 3.0)
        {
            if (srcPoints.GetLength(0) < 4 || dstPoints.GetLength(0) < 4)
            {
                throw new ArgumentException("At least 4 points are required for both source and destination.");
            }

            int numPoints = srcPoints.GetLength(0);
            double bestInlierCount = 0;
            double[,]? bestHomography = null;

            Random random = new();

            for (int iter = 0; iter < maxIterations; iter++)
            {
                // Step 1: Randomly select 4 points
                var indices = new HashSet<int>();
                while (indices.Count < 4)
                {
                    indices.Add(random.Next(numPoints));
                }

                // Step 2: Get the selected points
                double[,] selectedSrc = new double[4, 2];
                double[,] selectedDst = new double[4, 2];
                int i = 0;
                foreach (int idx in indices)
                {
                    selectedSrc[i, 0] = srcPoints[idx, 0];
                    selectedSrc[i, 1] = srcPoints[idx, 1];
                    selectedDst[i, 0] = dstPoints[idx, 0];
                    selectedDst[i, 1] = dstPoints[idx, 1];
                    i++;
                }

                // Step 3: Compute homography from the 4 selected points
                double[,] H = FindHomography(selectedSrc, selectedDst);

                // Step 4: Evaluate the homography on all points
                int inlierCount = 0;
                for (int j = 0; j < numPoints; j++)
                {
                    double[] srcPoint = [srcPoints[j, 0], srcPoints[j, 1], 1];
                    double[] dstPoint = [dstPoints[j, 0], dstPoints[j, 1], 1];

                    double[] transformedPoint = ApplyHomography(srcPoint, H);
                    double error = Math.Sqrt(Math.Pow(transformedPoint[0] - dstPoint[0], 2) + Math.Pow(transformedPoint[1] - dstPoint[1], 2));

                    if (error < threshold)
                    {
                        inlierCount++;
                    }
                }

                if (inlierCount > bestInlierCount)
                {
                    bestInlierCount = inlierCount;
                    bestHomography = H;
                }
            }

            return bestHomography;
        }

        [SupportedOSPlatform("windows")]
        public static Bitmap WarpImage(Bitmap inputImage, double[,] H, int outputWidth, int outputHeight)
        {
            double[,] H_inv = InverseHomography(H);
            Bitmap outputImage = new Bitmap(outputWidth, outputHeight);

            for (int y = 0; y < outputHeight; y++)
            {
                for (int x = 0; x < outputWidth; x++)
                {
                    double[] newPoints = [x, y];
                    double[] inputCoords = ApplyHomography(newPoints, H_inv);
                    // double[] inputCoords = ApplyHomography(x, y, H_inv);

                    int inputX = (int)Math.Floor(inputCoords[0]);
                    int inputY = (int)Math.Floor(inputCoords[1]);

                    if (inputX >= 0 && inputX < inputImage.Width && inputY >= 0 && inputY < inputImage.Height)
                    {
                        Color inputColor = BilinearInterpolation(inputImage, inputCoords[0], inputCoords[1]);

                        outputImage.SetPixel(x, y, inputColor);
                    }
                }
            }

            return outputImage;
        }

        private static double[] ApplyHomography(double[] point, double[,] H)
        {
            // Apply the homography matrix to the point
            double x = point[0];
            double y = point[1];
            double w = H[2, 0] * x + H[2, 1] * y + H[2, 2];

            double newX = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w;
            double newY = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w;

            return new double[] { newX, newY };
        }

        [SupportedOSPlatform("windows")]
        private static double[] ApplyHomography(int x, int y, double[,] H)
        {
            double[] result = new double[3];
            double[] point = [x, y, 1.0];// Homogeneous coordinates > (x, y, 1)


            for (int i = 0; i < 3; i++) // homography matrix * point
            {
                result[i] = 0;
                for (int j = 0; j < 3; j++)
                {
                    result[i] += H[i, j] * point[j];
                }
            }

            // Normalize
            double newX = result[0] / result[2];
            double newY = result[1] / result[2];

            return [newX, newY];
        }

        [SupportedOSPlatform("windows")]
        private static double[,] InverseHomography(double[,] H)
        {
            double[,] H_inv = new double[3, 3];
            double det = H[0, 0] * (H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1]) -
                         H[0, 1] * (H[1, 0] * H[2, 2] - H[1, 2] * H[2, 0]) +
                         H[0, 2] * (H[1, 0] * H[2, 1] - H[1, 1] * H[2, 0]);

            if (Math.Abs(det) < 1e-9)
            {
                throw new InvalidOperationException("Homography matrix cannot be invertible.");
            }

            double invDet = 1.0 / det;

            H_inv[0, 0] = (H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1]) * invDet;
            H_inv[0, 1] = (H[0, 2] * H[2, 1] - H[0, 1] * H[2, 2]) * invDet;
            H_inv[0, 2] = (H[0, 1] * H[1, 2] - H[0, 2] * H[1, 1]) * invDet;

            H_inv[1, 0] = (H[1, 2] * H[2, 0] - H[1, 0] * H[2, 2]) * invDet;
            H_inv[1, 1] = (H[0, 0] * H[2, 2] - H[0, 2] * H[2, 0]) * invDet;
            H_inv[1, 2] = (H[0, 2] * H[1, 0] - H[0, 0] * H[1, 2]) * invDet;

            H_inv[2, 0] = (H[1, 0] * H[2, 1] - H[1, 1] * H[2, 0]) * invDet;
            H_inv[2, 1] = (H[0, 1] * H[2, 0] - H[0, 0] * H[2, 1]) * invDet;
            H_inv[2, 2] = (H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]) * invDet;

            return H_inv;
        }

        [SupportedOSPlatform("windows")]
        private static Color BilinearInterpolation(Bitmap img, double x, double y)
        {
            double x1 = Math.Floor(x);
            double x2 = Math.Ceiling(x);
            double y1 = Math.Floor(y);
            double y2 = Math.Ceiling(y);

            // get corners > if coordinates are not valid, return black; else return color
            Color c11 = GetPixelColor(img, (int)x1, (int)y1);
            Color c12 = GetPixelColor(img, (int)x1, (int)y2);
            Color c21 = GetPixelColor(img, (int)x2, (int)y1);
            Color c22 = GetPixelColor(img, (int)x2, (int)y2);

            // bilinear interpolation
            double dx = x - x1;
            double dy = y - y1;

            int r = (int)((1 - dx) * (1 - dy) * c11.R + dx * (1 - dy) * c21.R + (1 - dx) * dy * c12.R + dx * dy * c22.R);
            int g = (int)((1 - dx) * (1 - dy) * c11.G + dx * (1 - dy) * c21.G + (1 - dx) * dy * c12.G + dx * dy * c22.G);
            int b = (int)((1 - dx) * (1 - dy) * c11.B + dx * (1 - dy) * c21.B + (1 - dx) * dy * c12.B + dx * dy * c22.B);

            return Color.FromArgb(r, g, b);
        }

        [SupportedOSPlatform("windows")]
        private static Color GetPixelColor(Bitmap img, int x, int y)
        {
            if (x < 0 || x >= img.Width || y < 0 || y >= img.Height)
            {
                return Color.Black;
            }

            return img.GetPixel(x, y);
        }
    }

}
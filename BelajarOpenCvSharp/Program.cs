using OpenCvSharp;

namespace BelajarOpenCvSharp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            using var camera = new VideoCapture(1);

            while (!camera.IsOpened()) continue;

            int width = (int)camera.Get(VideoCaptureProperties.FrameWidth);
            int height = (int)camera.Get(VideoCaptureProperties.FrameHeight);

            camera.AutoExposure = 1.0;


            Mat currentFrame = new(
                rows: height,
                cols: width,
                type: MatType.CV_8UC3,
                s: Scalar.Black
                );
            while (camera.Read(currentFrame))
            {
                if (currentFrame.Empty())
                {
                    continue;
                }

                //Cv2.CvtColor(currentFrame, currentFrame, ColorConversionCodes.BGR2GRAY);

                Cv2.ImShow("Video Feed", currentFrame);
                if (Cv2.WaitKey(1) == 27) break;
            }

            Cv2.DestroyAllWindows();
        }
    }
}

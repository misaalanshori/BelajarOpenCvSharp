using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BelajarOpenCvSharp
{
    public readonly record struct Anchor(float XCenter, float YCenter, float Width, float Height);

    public static class BlazePalmAnchors
    {
        public static List<Anchor> Generate(int inputSize = 192)
        {
            var anchors = new List<Anchor>(2016);
            int[] strides = [8, 16, 16, 16];
            int[] anchorsPerCell = [2, 6, 6, 6];

            for (int i = 0; i < strides.Length; i++)
            {
                int stride = strides[i];
                int count = anchorsPerCell[i];
                int gridSize = inputSize / stride;

                for (int y = 0; y < gridSize; y++)
                {
                    for (int x = 0; x < gridSize; x++)
                    {
                        float xCenter = (x + 0.5f) / gridSize;
                        float yCenter = (y + 0.5f) / gridSize;

                        for (int k = 0; k < count; k++)
                        {
                            anchors.Add(new Anchor(xCenter, yCenter, 1.0f, 1.0f));
                        }
                    }
                }
            }
            return anchors;
        }
    }
}

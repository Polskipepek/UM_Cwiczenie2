namespace UM_Cwiczenie2.SVMs {
    internal class SVMHelpers {
        public static double DotProduct(double[] a, double[] b) {
            double dotProduct = 0;
            for (int i = 0; i < a.Length; i++) {
                dotProduct += a[i] * b[i];
            }
            return dotProduct;
        }

        public static int Sign(double x) {
            return x >= 0 ? 1 : -1;
        }

        public static double[] CalculateKernelMatrix(double[][] inputs) {
            return new double[inputs.Length];
        }

        public static double[][] InitializeLagrangeMultipliers(double[][] inputs) {
            double[][] kernelMatrix = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++) {
                kernelMatrix[i] = new double[inputs.Length];
                for (int j = 0; j < inputs.Length; j++) {
                    kernelMatrix[i][j] = SVMHelpers.DotProduct(inputs[i], inputs[j]);
                }
            }

            return kernelMatrix;
        }
    }
}

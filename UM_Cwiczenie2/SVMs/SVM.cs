namespace UM_Cwiczenie2.SVMs {
    internal static class SVM {
        public static void Train(double[][] inputs, int[] outputs, double[] lagrangeMultipliers, double[][] kernelMatrix) {
            double tolerance = 0.001;
            double bias = 0;
            while (true) {
                int numChangedLagrangeMultipliers = 0;
                for (int i = 0; i < inputs.Length; i++) {
                    double prediction = SVMHelpers.DotProduct(
                        lagrangeMultipliers.Select((l) => l * outputs[i]).ToArray(),
                        kernelMatrix[i]) + bias;
                    double errorI = prediction - outputs[i];

                    if (CheckIfLagrangeMultiplierViolatesKKTConditions(outputs, lagrangeMultipliers, tolerance, i, errorI)) {
                        int j = SelectSecondLagrangeMultiplier(inputs, i);

                        double errorJ = CalculateEj(outputs, lagrangeMultipliers, kernelMatrix, bias, j);

                        double oldLagrangeMultiplierI = lagrangeMultipliers[i];
                        double oldLagrangeMultiplierJ = lagrangeMultipliers[j];

                        CalculateBounds(outputs, lagrangeMultipliers, i, j, out double L, out double H);

                        if (L == H) {
                            continue;
                        }

                        double eta = kernelMatrix[i][i] + kernelMatrix[j][j] - (2 * kernelMatrix[i][j]);

                        if (eta <= 0) {
                            continue;
                        }

                        lagrangeMultipliers[j] += outputs[j] * (errorI - errorJ) / eta;

                        // Clip Lagrange multiplier j
                        if (lagrangeMultipliers[j] > H) {
                            lagrangeMultipliers[j] = H;
                        } else if (lagrangeMultipliers[j] < L) {
                            lagrangeMultipliers[j] = j;
                        }

                        lagrangeMultipliers[i] += outputs[i] * outputs[j] * (oldLagrangeMultiplierJ - lagrangeMultipliers[j]);

                        //Dunno if delete
                        if (Math.Abs(oldLagrangeMultiplierJ - lagrangeMultipliers[j]) > 1e5) {
                            continue;
                        }

                        double biasI = bias - errorI - (outputs[i] * (lagrangeMultipliers[i] - oldLagrangeMultiplierI) * kernelMatrix[i][i]) - (outputs[j] * (lagrangeMultipliers[j] - oldLagrangeMultiplierJ) * kernelMatrix[i][j]);
                        double biasJ = bias - errorJ - (outputs[i] * (lagrangeMultipliers[i] - oldLagrangeMultiplierI) * kernelMatrix[i][j]) - (outputs[j] * (lagrangeMultipliers[j] - oldLagrangeMultiplierJ) * kernelMatrix[j][j]);

                        if (lagrangeMultipliers[i] > 0 && lagrangeMultipliers[i] < 1e6) {
                            bias = biasI;
                        } else if (lagrangeMultipliers[j] > 0 && lagrangeMultipliers[j] < 1e6) {
                            bias = biasJ;
                        } else {
                            bias = (biasI + biasJ) / 2;
                        }

                        numChangedLagrangeMultipliers++;
                    }

                    if (numChangedLagrangeMultipliers == 0) {
                        break;
                    }
                }

                double[] weights = CalculateWeights(inputs, outputs, lagrangeMultipliers);

                List<double[]> supportVectors = new();
                List<int> supportVectorOutputs = new();
                CalculateSupportVectors(inputs, outputs, lagrangeMultipliers, supportVectors, supportVectorOutputs);

                Predict(bias, weights);
                break;
            }
        }

        private static void Predict(double bias, double[] weights) {
            double[] testInput1 = new double[] { -1, -1 };
            double prediction1 = SVMHelpers.Sign(SVMHelpers.DotProduct(weights, testInput1) + bias);
            Console.WriteLine("Prediction for (-1, -1): " + prediction1);

            double[] testInput2 = new double[] { -1, 1 };
            double prediction2 = SVMHelpers.Sign(SVMHelpers.DotProduct(weights, testInput2) + bias);
            Console.WriteLine("Prediction for (-1, 1): " + prediction2);

            double[] testInput3 = new double[] { 1, -1 };
            double prediction3 = SVMHelpers.Sign(SVMHelpers.DotProduct(weights, testInput3) + bias);
            Console.WriteLine("Prediction for (1, -1): " + prediction3);

            double[] testInput4 = new double[] { 1, 1 };
            double prediction4 = SVMHelpers.Sign(SVMHelpers.DotProduct(weights, testInput4) + bias);
            Console.WriteLine("Prediction for (1, 1): " + prediction4);
        }

        private static void CalculateSupportVectors(double[][] inputs, int[] outputs, double[] lagrangeMultipliers, List<double[]> supportVectors, List<int> supportVectorOutputs) {
            for (int i = 0; i < inputs.Length; i++) {
                if (lagrangeMultipliers[i] > 0) {
                    supportVectors.Add(inputs[i]);
                    supportVectorOutputs.Add(outputs[i]);
                }
            }
        }

        private static double[] CalculateWeights(double[][] inputs, int[] outputs, double[] lagrangeMultipliers) {
            double[] weights = new double[inputs[0].Length];
            for (int i = 0; i < inputs.Length; i++) {
                for (int j = 0; j < weights.Length; j++) {
                    weights[j] += lagrangeMultipliers[i] * outputs[i] * inputs[i][j];
                }
            }

            return weights;
        }

        private static void CalculateBounds(int[] outputs, double[] lagrangeMultipliers, int i, int j, out double L, out double H) {
            if (outputs[i] != outputs[j]) {
                L = Math.Max(0, lagrangeMultipliers[j] - lagrangeMultipliers[i]);
                H = Math.Min(1e6, 1e6 + lagrangeMultipliers[j] - lagrangeMultipliers[i]);
            } else {
                L = Math.Max(0, lagrangeMultipliers[j] + lagrangeMultipliers[i] - 1e6);
                H = Math.Min(1e6, lagrangeMultipliers[j] + lagrangeMultipliers[i]);
            }
        }

        private static double CalculateEj(int[] outputs, double[] lagrangeMultipliers, double[][] kernelMatrix, double bias, int j) {
            double predictionJ = SVMHelpers.DotProduct(
                lagrangeMultipliers.Select((l) => l * outputs[j]).ToArray(),
                kernelMatrix[j]) + bias;
            double errorJ = predictionJ - outputs[j];
            return errorJ;
        }


        private static int SelectSecondLagrangeMultiplier(double[][] inputs, int i) {
            int j;
            do {
                j = new Random().Next(inputs.Length);
            } while (j == i);
            return j;
        }

        private static bool CheckIfLagrangeMultiplierViolatesKKTConditions(int[] outputs, double[] lagrangeMultipliers, double tolerance, int i, double error) {
            return (outputs[i] * error < -tolerance && lagrangeMultipliers[i] < 1e6) ||
                                    (outputs[i] * error > tolerance && lagrangeMultipliers[i] > 0);
        }
    }
}

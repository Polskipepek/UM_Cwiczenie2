using UM_Cwiczenie2.SVMs;

Console.WriteLine("Hello, World!");

//1. Prepare data.
double[][] inputs;
int[] outputs;

PrepareData(out inputs, out outputs);

//inputs = DataReader.ReadData($"{Environment.CurrentDirectory}/Data/TrainData.csv", ";"));

//2. Calculate Kernel Matrix.
double[] lagrangeMultipliers = SVMHelpers.CalculateKernelMatrix(inputs);

//3. Initialize Lagrange multipliers.
double[][] kernelMatrix = SVMHelpers.InitializeLagrangeMultipliers(inputs);

//4. Train model using Kernel Matrix and Lagrange multipliers.
SVM.Train(inputs, outputs, lagrangeMultipliers, kernelMatrix);


static void PrepareData(out double[][] inputs, out int[] outputs) {
    inputs = new double[][] { new double[] { -1, -1 }, new double[] { -1, 1 }, new double[] { 1, -1 }, new double[] { 1, 1 } };
    outputs = new int[] { -1, -1, -1, 1 };
}


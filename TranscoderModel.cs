using System;
using System.IO;
using MLHelpers;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace VideoTranscoderTimePredictor
{
    public class TranscoderModel : IMLTrainer
    {
        //Base path of the application
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        //Path to the data file
        private static string _dataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "transcoding_mesurments.csv");
        //Path to where the model will be saved
        public static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", typeof(TranscoderModel)+".zip");
        
        //Reference to the MLContext
        private MLContext _mlContext;
		//Reference to the pipeline of the model
        private IEstimator<ITransformer> _pipeline;
        //Reference to the model    
        private ITransformer _model;
        //Training data
        IDataView _trainData;
        //Testing data
        IDataView _testData;
        
        public TranscoderModel()
        {
            _mlContext = new MLContext();
        }

        //Loads the data
        public void LoadData()
        {
            //Read all the data
            IDataView allData = _mlContext.Data.LoadFromTextFile<TranscoderData>(path: _dataPath, hasHeader: true, separatorChar: ',');

            //split the data into test and training
            DataOperationsCatalog.TrainTestData splitData = _mlContext.Data.TrainTestSplit(allData, testFraction: 0.3,seed:1);
            _trainData = splitData.TrainSet;
            _testData = splitData.TestSet;  
        }

        //Data pre processing 
        public void BuildPipeline()
        {

        }


        //Build and train the model
        public void BuildAndTrainModel()
        {

        }


        //Evaluate the performance of the model
        public void EvaluateModel() 
        {
            var predictions = _model.Transform(_testData);
            var metrics = _mlContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        //Save the model to file
        public void SaveModelToFile(string pathToFile)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                _mlContext.Model.Save(_model,_trainData.Schema, fs);
            }
        }
    }    
}

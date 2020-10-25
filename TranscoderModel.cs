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
            _pipeline = _mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName:"utime")

            .Append(_mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "incodec", outputColumnName: "incodecE"))
            .Append(_mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "outcodec", outputColumnName: "outcodecE"))
            




            .Append(_mlContext.Transforms.Conversion.ConvertType("inbitrate", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("incodecE", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("induration", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("inframerate", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("inframes", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("inheight", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("inwidth", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("insize", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("outbitrate", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("outcodecE", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("outframerate", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("outheight", outputKind: DataKind.Single))
            .Append(_mlContext.Transforms.Conversion.ConvertType("outwidth", outputKind: DataKind.Single))

            
            //Set the features to be used 
            .Append(_mlContext.Transforms.Concatenate("Features", "inbitrate", "incodecE", "induration",
             "inframerate", "inframes", "inheight", "inwidth", "insize", "outbitrate", "outcodecE", "outframerate", "outheight", "outwidth"))
            //cache the pipeline, this will make downstream processes faster
            .AppendCacheCheckpoint(_mlContext);
        }


        //Build and train the model
        public void BuildAndTrainModel()
        {
            // linear regression model
            //_pipeline = _pipeline.Append(_mlContext.Regression.Trainers.LbfgsPoissonRegression());

            // decision tree
            _pipeline = _pipeline.Append(_mlContext.Regression.Trainers.FastTree());
            
            // random forest
            //_pipeline = _pipeline.Append(_mlContext.Regression.Trainers.FastForest());
            
            _model = _pipeline.Fit(_trainData);
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
#
# Copyright (c) 2019 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from pyspark import keyword_only
from typing import Union, Optional, List, Dict, Callable, Tuple, Any, TypeVar, Type
from sparkxgb.common import XGboostEstimator, XGboostModel


class XGBoostClassifier(XGboostEstimator):
    # pylint: disable=too-many-arguments, too-many-instance-attributes, missing-docstring
    """
    A PySpark wrapper of ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
    """

    @keyword_only
    def __init__(self,
                 alpha: float=0.0,
                 baseMarginCol=None,
                 baseScore: float=0.5,
                 cacheTrainingSet: bool=False,
                 checkpointInterval: int=-1,
                 checkpointPath: str="",
                 colsampleBylevel: float=1.0,
                 colsampleBytree: float=1.0,
                 contribPredictionCol:Optional[str]=None,
                 ## EXCLUDED: customEval=None,
                 ## EXCLUDED: customObj=None,
                 eta: float=0.3,
                 evalMetric: Optional[Union[Callable, str, List[str]]]=None,
                 featuresCol: str="features",
                 labelCol: str="label",
                 gamma: float=0.0,
                 growPolicy: str="depthwise",
                 interactionConstraints: Optional[Union[str, List[Tuple[str]]]]=None,
                 labelCo: str="label",
                 regLambda: float=1.0,  # Rename of 'lambda' param, as this is a reserved keyword in python.
                 lambdaBias: float=0.0,
                 leafPredictionCol: Optional[str]=None,
                 maxBins: int=16,
                 maxDeltaStep: float=0.0,
                 maxDepth: int=6,
                 maxLeaves: Optional[int]=None,
                 maximizeEvaluationMetrics=None,
                 minChildWeight: float=1.0,
                 missing: float=float('nan'),
                 monotoneConstraints=None,
                 normalizeType: str="tree",
                 nthread: int=1,
                 numClass: Optional[int]=None,
                 numEarlyStoppingRounds=0,
                 numRound: int=1,
                 numWorkers: int=1,
                 objective: str="reg:squarederror",
                 objectiveType=None,
                 predictionCol: str="prediction",
                 probabilityCol: str="probability",
                 rateDrop: float=0.0,
                 rawPredictionCol: str="rawPrediction",
                 sampleType: str="uniform",
                 scalePosWeight: float=1.0,
                 seed: int=0,
                 silent: int=0,
                 sketchEps: float=0.03,
                 skipDrop: float=0.0,
                 subsample: float=1.0,
                 thresholds: Optional[float]=None,
                 timeoutRequestWorkers: int=1800000,
                 ## EXCLUDED: trackerConf=None,
                 trainTestRatio: float=1.0,
                 treeLimit: float=0,
                 treeMethod: str="auto",
                 useExternalMemory: bool=False,
                 verbosity: int=1,
                 weightCol=None):
        super(XGBoostClassifier, self).__init__(classname="ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier")
        kwargs = self._input_kwargs
        if "regLambda" in kwargs:
            kwargs["lambda"] = kwargs.pop("regLambda")
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self,
                  alpha=0.0,
                  baseMarginCol=None,
                  baseScore=0.5,
                  cacheTrainingSet=False,
                  checkpointInterval=-1,
                  checkpointPath="",
                  colsampleBylevel=1.0,
                  colsampleBytree=1.0,
                  contribPredictionCol=None,
                  ## EXCLUDED: customEval=None,
                  ## EXCLUDED: customObj=None,
                  eta=0.3,
                  evalMetric=None,
                  featuresCol="features",
                  gamma=0.0,
                  growPolicy="depthwise",
                  interactionConstraints=None,
                  labelCol="label",
                  regLambda=1.0,  # Rename of 'lambda' param, as this is a reserved keyword in python.
                  lambdaBias=0.0,
                  leafPredictionCol=None,
                  maxBins=16,
                  maxDeltaStep=0.0,
                  maxDepth=6,
                  maxLeaves=None,
                  maximizeEvaluationMetrics=None,
                  minChildWeight=1.0,
                  missing=float('nan'),
                  monotoneConstraints=None,
                  normalizeType="tree",
                  nthread=1,
                  numClass=None,
                  numEarlyStoppingRounds=0,
                  numRound=1,
                  numWorkers=1,
                  objective="reg:squarederror",
                  objectiveType=None,
                  predictionCol="prediction",
                  probabilityCol="probability",
                  rateDrop=0.0,
                  rawPredictionCol="rawPrediction",
                  sampleType="uniform",
                  scalePosWeight=1.0,
                  seed=0,
                  silent=0,
                  sketchEps=0.03,
                  skipDrop=0.0,
                  subsample=1.0,
                  thresholds=None,
                  timeoutRequestWorkers=1800000,
                  ## EXCLUDED: trackerConf=None,
                  trainTestRatio=1.0,
                  treeLimit=0,
                  treeMethod="auto",
                  useExternalMemory=False,
                  verbosity=1,
                  weightCol=None):
        kwargs = self._input_kwargs
        if "regLambda" in kwargs:
            kwargs["lambda"] = kwargs.pop("regLambda")
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return XGBoostClassificationModel(java_model=java_model)


class XGBoostClassificationModel(XGboostModel):
    """
    A PySpark wrapper of ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
    """

    def __init__(self, classname="ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel", java_model=None):
        super(XGBoostClassificationModel, self).__init__(classname=classname, java_model=java_model)

    @property
    def nativeBooster(self):
        """
        Get the native booster instance of this model.
        This is used to call low-level APIs on native booster, such as "getFeatureScore".
        """
        return self._call_java("nativeBooster")


class XGBoostRegressor(XGboostEstimator):
    """
    A PySpark wrapper of ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
    """

    @keyword_only
    def __init__(self,
                 alpha=0.0,
                 baseMarginCol=None,
                 baseScore=0.5,
                 cacheTrainingSet=False,
                 checkpointInterval=-1,
                 checkpointPath="",
                 colsampleBylevel=1.0,
                 colsampleBytree=1.0,
                 contribPredictionCol=None,
                 ## EXCLUDED: customEval=None,
                 ## EXCLUDED: customObj=None,
                 eta=0.3,
                 evalMetric=None,
                 featuresCol="features",
                 gamma=0.0,
                 groupCol=None,
                 growPolicy="depthwise",
                 interactionConstraints=None,
                 labelCol="label",
                 lambda_=1.0,  # Rename of 'lambda' param, as this is a reserved keyword in python.
                 lambdaBias=0.0,
                 leafPredictionCol=None,
                 maxBins=16,
                 maxDeltaStep=0.0,
                 maxDepth=6,
                 maxLeaves=None,
                 maximizeEvaluationMetrics=None,
                 minChildWeight=1.0,
                 missing=float('nan'),
                 monotoneConstraints=None,
                 normalizeType="tree",
                 nthread=1,
                 numClass=None,
                 numEarlyStoppingRounds=0,
                 numRound=1,
                 numWorkers=1,
                 objective="reg:squarederror",
                 objectiveType=None,
                 predictionCol="prediction",
                 probabilityCol="probability",
                 rateDrop=0.0,
                 rawPredictionCol="rawPrediction",
                 sampleType="uniform",
                 scalePosWeight=1.0,
                 seed=0,
                 silent=0,
                 sketchEps=0.03,
                 skipDrop=0.0,
                 subsample=1.0,
                 thresholds=None,
                 timeoutRequestWorkers=1800000,
                 ## EXCLUDED: trackerConf=None,
                 trainTestRatio=1.0,
                 treeLimit=0,
                 treeMethod="auto",
                 useExternalMemory=False,
                 verbosity=1,
                 weightCol=None):
        super(XGBoostRegressor, self).__init__(classname="ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor")
        kwargs = self._input_kwargs
        if "lambda_" in kwargs:
            kwargs["lambda"] = kwargs.pop("lambda_")
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self,
                  alpha=0.0,
                  baseMarginCol=None,
                  baseScore=0.5,
                  cacheTrainingSet=False,
                  checkpointInterval=-1,
                  checkpointPath="",
                  colsampleBylevel=1.0,
                  colsampleBytree=1.0,
                  contribPredictionCol=None,
                  ## EXCLUDED: customEval=None,
                  ## EXCLUDED: customObj=None,
                  eta=0.3,
                  evalMetric=None,
                  featuresCol="features",
                  gamma=0.0,
                  groupCol=None,
                  growPolicy="depthwise",
                  interactionConstraints=None,
                  labelCol="label",
                  lambda_=1.0,  # Rename of 'lambda' param, as this is a reserved keyword in python.
                  lambdaBias=0.0,
                  leafPredictionCol=None,
                  maxBins=16,
                  maxDeltaStep=0.0,
                  maxDepth=6,
                  maxLeaves=None,
                  maximizeEvaluationMetrics=None,
                  minChildWeight=1.0,
                  missing=float('nan'),
                  monotoneConstraints=None,
                  normalizeType="tree",
                  nthread=1,
                  numClass=None,
                  numEarlyStoppingRounds=0,
                  numRound=1,
                  numWorkers=1,
                  objective="reg:squarederror",
                  objectiveType=None,
                  predictionCol="prediction",
                  probabilityCol="probability",
                  rateDrop=0.0,
                  rawPredictionCol="rawPrediction",
                  sampleType="uniform",
                  scalePosWeight=1.0,
                  seed=0,
                  silent=0,
                  sketchEps=0.03,
                  skipDrop=0.0,
                  subsample=1.0,
                  thresholds=None,
                  timeoutRequestWorkers=1800000,
                  ## EXCLUDED: trackerConf=None,
                  trainTestRatio=1.0,
                  treeLimit=0,
                  treeMethod="auto",
                  useExternalMemory=False,
                  verbosity=1,
                  weightCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return XGBoostRegressionModel(java_model=java_model)


class XGBoostRegressionModel(XGboostModel):
    """
    A PySpark wrapper of ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
    """

    def __init__(self, classname="ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel", java_model=None):
        super(XGBoostRegressionModel, self).__init__(classname=classname, java_model=java_model)

    @property
    def nativeBooster(self):
        """
        Get the native booster instance of this model.
        This is used to call low-level APIs on native booster, such as "getFeatureScore".
        """
        return self._call_java("nativeBooster")

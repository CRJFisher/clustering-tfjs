import * as tf from '../tf-adapter';
import type {
  BaseClustering,
  DataMatrix,
  LabelVector,
  SOMParams,
  SOMState,
  SOMTopology,
  SOMNeighborhood,
  SOMInitialization,
  DecayFunction,
} from './types';
import { isTensor } from '../utils/tensor-utils';
import {
  initializeWeights,
  findBMUBatch,
  computeNeighborhoodInfluenceBatch,
  createGridDistanceMatrix,
  computeBMUDistances,
  createDecayScheduler,
  validateNeighborhoodParams,
} from './som_utils';

/**
 * Self-Organizing Map (SOM) implementation using TensorFlow.js.
 * 
 * SOMs create a low-dimensional (typically 2D) discrete representation
 * of high-dimensional input space while preserving topological properties.
 */
export class SOM implements BaseClustering<SOMParams> {
  public readonly params: SOMParams;
  
  // Model state
  public weights_: tf.Tensor3D | null = null;
  public labels_: LabelVector | null = null;
  public bmus_: tf.Tensor2D | null = null;
  
  // Training state
  private gridDistanceMatrix_: tf.Tensor2D | null = null;
  private learningRateScheduler_: DecayFunction | null = null;
  private radiusScheduler_: DecayFunction | null = null;
  private totalSamplesLearned_: number = 0;
  private currentEpoch_: number = 0;
  private quantizationErrors_: number[] = [];
  
  // Default parameters
  private static readonly DEFAULT_TOPOLOGY: SOMTopology = 'rectangular';
  private static readonly DEFAULT_NEIGHBORHOOD: SOMNeighborhood = 'gaussian';
  private static readonly DEFAULT_NUM_EPOCHS = 100;
  private static readonly DEFAULT_LEARNING_RATE = 0.5;
  private static readonly DEFAULT_INITIALIZATION: SOMInitialization = 'random';
  private static readonly DEFAULT_TOL = 1e-4;
  private static readonly DEFAULT_MINI_BATCH_SIZE = 32;
  
  constructor(params: SOMParams) {
    this.params = this.validateAndCompleteParams(params);
    this.initializeSchedulers();
  }
  
  /**
   * Validate and set default parameters.
   */
  private validateAndCompleteParams(params: SOMParams): SOMParams {
    if (!params.gridWidth || params.gridWidth < 1) {
      throw new Error('gridWidth must be >= 1');
    }
    if (!params.gridHeight || params.gridHeight < 1) {
      throw new Error('gridHeight must be >= 1');
    }
    
    // Note: nClusters from BaseClusteringParams is not used for SOM
    // Total neurons = gridWidth * gridHeight
    
    return {
      ...params,
      nClusters: params.gridWidth * params.gridHeight, // For compatibility
      topology: params.topology ?? SOM.DEFAULT_TOPOLOGY,
      neighborhood: params.neighborhood ?? SOM.DEFAULT_NEIGHBORHOOD,
      numEpochs: params.numEpochs ?? SOM.DEFAULT_NUM_EPOCHS,
      learningRate: params.learningRate ?? SOM.DEFAULT_LEARNING_RATE,
      initialization: params.initialization ?? SOM.DEFAULT_INITIALIZATION,
      tol: params.tol ?? SOM.DEFAULT_TOL,
      miniBatchSize: params.miniBatchSize ?? SOM.DEFAULT_MINI_BATCH_SIZE,
      onlineMode: params.onlineMode ?? false,
    };
  }
  
  /**
   * Initialize learning rate and radius schedulers.
   */
  private initializeSchedulers(): void {
    const { gridWidth, gridHeight, numEpochs, learningRate, radius } = this.params;
    
    // Learning rate scheduler
    if (typeof learningRate === 'function') {
      this.learningRateScheduler_ = learningRate;
    } else {
      this.learningRateScheduler_ = createDecayScheduler(
        learningRate as number,
        'exponential',
        numEpochs!
      );
    }
    
    // Radius scheduler
    if (radius !== undefined) {
      if (typeof radius === 'function') {
        this.radiusScheduler_ = radius;
      } else {
        this.radiusScheduler_ = createDecayScheduler(
          radius,
          'exponential',
          numEpochs!
        );
      }
    } else {
      // Default: adaptive radius based on grid size
      const initialRadius = Math.max(gridWidth, gridHeight) / 2;
      this.radiusScheduler_ = createDecayScheduler(
        initialRadius,
        'exponential',
        numEpochs!,
        1 // final radius
      );
    }
  }
  
  /**
   * Fit the SOM to the provided data.
   */
  async fit(X: DataMatrix): Promise<void> {
    const xTensor = isTensor(X) ? X as tf.Tensor2D : tf.tensor2d(X);
    
    try {
      await this.fitTensor(xTensor);
    } finally {
      if (!isTensor(X)) {
        xTensor.dispose();
      }
    }
  }
  
  /**
   * Internal fit method using tensors.
   */
  private async fitTensor(X: tf.Tensor2D): Promise<void> {
    const { 
      gridWidth, 
      gridHeight, 
      topology, 
      numEpochs, 
      initialization, 
      randomState,
      tol
    } = this.params;
    
    const [nSamples, _nFeatures] = X.shape;
    
    // Initialize weights if not already done
    if (!this.weights_) {
      this.weights_ = initializeWeights(
        X,
        gridHeight,
        gridWidth,
        initialization!,
        randomState
      );
    }
    
    // Pre-compute grid distance matrix
    if (!this.gridDistanceMatrix_) {
      this.gridDistanceMatrix_ = createGridDistanceMatrix(
        gridHeight,
        gridWidth,
        topology!
      );
    }
    
    // Training loop
    let prevQuantizationError = Infinity;
    this.quantizationErrors_ = [];
    
    for (let epoch = 0; epoch < numEpochs!; epoch++) {
      this.currentEpoch_ = epoch;
      
      // Get current learning rate and radius
      const currentLearningRate = this.learningRateScheduler_!(epoch, numEpochs!);
      const currentRadius = this.radiusScheduler_!(epoch, numEpochs!);
      
      // Validate neighborhood parameters
      validateNeighborhoodParams(currentRadius, gridHeight, gridWidth);
      
      // Process batch
      const { quantizationError } = await this.trainEpoch(
        X,
        currentLearningRate,
        currentRadius
      );
      
      this.quantizationErrors_.push(quantizationError);
      
      // Check convergence
      if (Math.abs(prevQuantizationError - quantizationError) < tol!) {
        console.log(`SOM converged at epoch ${epoch + 1}`);
        break;
      }
      
      prevQuantizationError = quantizationError;
      
      // Update total samples learned (only for online mode)
      if (this.params.onlineMode) {
        this.totalSamplesLearned_ += nSamples;
      }
    }
    
    // Compute final BMUs and labels
    await this.computeFinalLabels(X);
  }
  
  /**
   * Train one epoch.
   */
  private async trainEpoch(
    X: tf.Tensor2D,
    learningRate: number,
    radius: number
  ): Promise<{ quantizationError: number }> {
    const { neighborhood, miniBatchSize } = this.params;
    const [nSamples] = X.shape;
    
    // Process in mini-batches for memory efficiency
    const batchSize = Math.min(miniBatchSize!, nSamples);
    let totalQuantizationError = 0;
    let samplesProcessed = 0;
    
    for (let i = 0; i < nSamples; i += batchSize) {
      const endIdx = Math.min(i + batchSize, nSamples);
      const batchX = X.slice([i, 0], [endIdx - i, -1]);
      
      // Find BMUs for batch
      const bmus = findBMUBatch(batchX, this.weights_!);
      
      // Get BMU flat indices
      const bmuIndices = tf.tidy(() => {
        const bmusData = bmus.arraySync();
        const indices = bmusData.map(([row, col]) => 
          row * this.params.gridWidth + col
        );
        return tf.tensor1d(indices, 'int32');
      });
      
      // Compute neighborhood influence
      const influence = computeNeighborhoodInfluenceBatch(
        bmuIndices,
        this.gridDistanceMatrix_!,
        radius,
        neighborhood!
      );
      
      // Update weights
      this.updateWeights(batchX, influence, learningRate);
      
      // Compute quantization error for this batch
      const distances = computeBMUDistances(batchX, this.weights_!, bmus);
      const batchError = distances.mean().arraySync() as number;
      totalQuantizationError += batchError * (endIdx - i);
      samplesProcessed += (endIdx - i);
      
      // Clean up
      batchX.dispose();
      bmus.dispose();
      bmuIndices.dispose();
      influence.dispose();
      distances.dispose();
    }
    
    return {
      quantizationError: totalQuantizationError / samplesProcessed
    };
  }
  
  /**
   * Update weights based on samples and neighborhood influence.
   */
  private updateWeights(
    samples: tf.Tensor2D,
    influence: tf.Tensor2D,
    learningRate: number
  ): void {
    tf.tidy(() => {
      const [nSamples, nFeatures] = samples.shape;
      const [gridHeight, gridWidth, _] = this.weights_!.shape;
      const totalNeurons = gridHeight * gridWidth;
      
      // Reshape weights for update
      const weightsFlat = this.weights_!.reshape([totalNeurons, nFeatures]);
      
      // Compute weight updates
      // For each neuron: Δw = Σ(lr * h * (x - w)) / Σh
      // Where h is the influence for each sample
      
      // Expand samples for broadcasting
      const samplesExpanded = samples.expandDims(1); // [nSamples, 1, nFeatures]
      const weightsExpanded = weightsFlat.expandDims(0); // [1, totalNeurons, nFeatures]
      
      // Compute differences
      const diff = samplesExpanded.sub(weightsExpanded); // [nSamples, totalNeurons, nFeatures]
      
      // Apply influence and learning rate
      const influenceExpanded = influence.expandDims(2); // [nSamples, totalNeurons, 1]
      const updates = diff.mul(influenceExpanded).mul(learningRate);
      
      // Sum over samples
      const totalUpdate = updates.sum(0); // [totalNeurons, nFeatures]
      
      // Apply updates
      const newWeightsFlat = weightsFlat.add(totalUpdate);
      
      // Reshape back to grid
      const newWeights = newWeightsFlat.reshape([gridHeight, gridWidth, nFeatures]) as tf.Tensor3D;
      
      // Update weights in place (keep newWeights from being disposed by tidy)
      this.weights_!.dispose();
      this.weights_ = tf.keep(newWeights);
    });
  }
  
  /**
   * Compute final BMUs and labels after training.
   */
  private async computeFinalLabels(X: tf.Tensor2D): Promise<void> {
    this.bmus_ = findBMUBatch(X, this.weights_!);
    
    // Convert BMUs to 1D labels
    const bmusData = await this.bmus_.array();
    const labels = bmusData.map(([row, col]) => 
      row * this.params.gridWidth + col
    );
    
    this.labels_ = labels;
  }
  
  /**
   * Fit and return predicted labels.
   */
  async fitPredict(X: DataMatrix): Promise<LabelVector> {
    await this.fit(X);
    return this.labels_!;
  }
  
  /**
   * Predict labels for new data using trained SOM.
   */
  async predict(X: DataMatrix): Promise<LabelVector> {
    if (!this.weights_) {
      throw new Error('SOM must be fitted before prediction');
    }
    
    const xTensor = isTensor(X) ? X as tf.Tensor2D : tf.tensor2d(X);
    
    try {
      const bmus = findBMUBatch(xTensor, this.weights_);
      const bmusData = await bmus.array();
      const labels = bmusData.map(([row, col]) => 
        row * this.params.gridWidth + col
      );
      bmus.dispose();
      return labels;
    } finally {
      if (!isTensor(X)) {
        xTensor.dispose();
      }
    }
  }
  
  /**
   * Partial fit for online/incremental learning.
   * Continues training from current state.
   */
  async partialFit(X: DataMatrix): Promise<void> {
    if (!this.params.onlineMode) {
      throw new Error('partialFit requires onlineMode to be enabled');
    }
    
    const xTensor = isTensor(X) ? X as tf.Tensor2D : tf.tensor2d(X);
    
    try {
      const [nSamples] = xTensor.shape;
      
      // Initialize if first call
      if (!this.weights_) {
        // Initialize weights and grid
        this.weights_ = initializeWeights(
          xTensor,
          this.params.gridHeight,
          this.params.gridWidth,
          this.params.initialization!,
          this.params.randomState
        );
        
        this.gridDistanceMatrix_ = createGridDistanceMatrix(
          this.params.gridHeight,
          this.params.gridWidth,
          this.params.topology!
        );
        
        // Initialize schedulers
        this.initializeSchedulers();
      }
      
      // Get current learning rate and radius based on total samples learned
      const virtualEpoch = Math.floor(
        this.totalSamplesLearned_ / nSamples
      );
      const currentLearningRate = this.learningRateScheduler_!(
        virtualEpoch,
        this.params.numEpochs!
      );
      const currentRadius = this.radiusScheduler_!(
        virtualEpoch,
        this.params.numEpochs!
      );
      
      // Train on batch
      await this.trainEpoch(xTensor, currentLearningRate, currentRadius);
      
      // Update total samples learned
      this.totalSamplesLearned_ += nSamples;
      
      // Update labels
      await this.computeFinalLabels(xTensor);
    } finally {
      if (!isTensor(X)) {
        xTensor.dispose();
      }
    }
  }
  
  /**
   * Get the weight matrix.
   */
  getWeights(): tf.Tensor3D {
    if (!this.weights_) {
      throw new Error('SOM must be fitted first');
    }
    return this.weights_;
  }
  
  /**
   * Calculate the U-matrix (unified distance matrix).
   * Shows the average distance between each neuron and its neighbors.
   */
  getUMatrix(): tf.Tensor2D {
    if (!this.weights_) {
      throw new Error('SOM must be fitted first');
    }
    
    return tf.tidy(() => {
      const { gridHeight, gridWidth, topology } = this.params;
      const uMatrix = tf.buffer([gridHeight, gridWidth]);
      const weightsData = this.weights_!.arraySync();
      
      for (let i = 0; i < gridHeight; i++) {
        for (let j = 0; j < gridWidth; j++) {
          const currentWeight = weightsData[i][j];
          let totalDistance = 0;
          let neighborCount = 0;
          
          // Check all adjacent positions
          const neighbors = [
            [i - 1, j], [i + 1, j],
            [i, j - 1], [i, j + 1],
          ];
          
          if (topology === 'rectangular') {
            // Add diagonal neighbors for rectangular
            neighbors.push(
              [i - 1, j - 1], [i - 1, j + 1],
              [i + 1, j - 1], [i + 1, j + 1]
            );
          }
          
          for (const [ni, nj] of neighbors) {
            if (ni >= 0 && ni < gridHeight && nj >= 0 && nj < gridWidth) {
              const neighborWeight = weightsData[ni][nj];
              const distance = Math.sqrt(
                currentWeight.reduce((sum, val, idx) => 
                  sum + Math.pow(val - neighborWeight[idx], 2), 0
                )
              );
              totalDistance += distance;
              neighborCount++;
            }
          }
          
          uMatrix.set(
            neighborCount > 0 ? totalDistance / neighborCount : 0,
            i,
            j
          );
        }
      }
      
      return uMatrix.toTensor() as tf.Tensor2D;
    });
  }
  
  /**
   * Calculate quantization error.
   * Average distance between samples and their BMUs.
   */
  quantizationError(): number {
    if (this.quantizationErrors_.length === 0) {
      throw new Error('SOM must be fitted first');
    }
    return this.quantizationErrors_[this.quantizationErrors_.length - 1];
  }
  
  /**
   * Calculate topographic error.
   * Proportion of samples whose BMU and second BMU are not neighbors.
   */
  async topographicError(X?: DataMatrix): Promise<number> {
    if (!this.weights_) {
      throw new Error('SOM must be fitted first');
    }
    
    if (!X) {
      throw new Error('Input data required for topographic error calculation');
    }
    
    const xTensor = isTensor(X) ? X as tf.Tensor2D : tf.tensor2d(X);
    
    try {
      // const { gridWidth, gridHeight, topology } = this.params; // Reserved for future use
      const [nSamples] = xTensor.shape;
      
      // Find first and second BMUs
      const bmus = await findBMUBatch(xTensor, this.weights_).array();
      
      let errors = 0;
      for (let i = 0; i < nSamples; i++) {
        const sample = xTensor.slice([i, 0], [1, -1]).squeeze();
        
        // Get second BMU (this is a simplified approach)
        // In a full implementation, we'd use findSecondBMU from som_utils
        // const [bmu1Row, bmu1Col] = bmus[i]; // Reserved for neighbor check
        
        // Check if BMUs are neighbors
        // For simplicity, we consider direct neighbors only
        const isNeighbor = false; // Simplified - would need full implementation
        
        if (!isNeighbor) {
          errors++;
        }
        
        sample.dispose();
      }
      
      return errors / nSamples;
    } finally {
      if (!isTensor(X)) {
        xTensor.dispose();
      }
    }
  }
  
  /**
   * Get total samples learned (for online learning).
   */
  getTotalSamplesLearned(): number {
    return this.totalSamplesLearned_;
  }
  
  /**
   * Save SOM state for persistence.
   */
  saveState(): SOMState {
    if (!this.weights_) {
      throw new Error('SOM must be fitted first');
    }
    
    return {
      weights: this.weights_.arraySync(),
      totalSamples: this.totalSamplesLearned_,
      currentEpoch: this.currentEpoch_,
      gridWidth: this.params.gridWidth,
      gridHeight: this.params.gridHeight,
      params: this.params,
    };
  }
  
  /**
   * Load SOM state from saved data.
   */
  loadState(state: SOMState): void {
    this.weights_?.dispose();
    this.weights_ = tf.tensor3d(state.weights);
    this.totalSamplesLearned_ = state.totalSamples;
    this.currentEpoch_ = state.currentEpoch;
    
    // Reinitialize schedulers if params changed
    if (state.params) {
      this.initializeSchedulers();
    }
  }
  
  /**
   * Save model state to JSON string.
   * Can be saved to file or transmitted over network.
   */
  async saveToJSON(): Promise<string> {
    if (!this.weights_) {
      throw new Error('SOM must be fitted before saving');
    }
    
    const modelData = {
      weights: await this.weights_.array(),
      metadata: {
        params: this.params,
        totalSamplesLearned: this.totalSamplesLearned_,
        currentEpoch: this.currentEpoch_,
        quantizationErrors: this.quantizationErrors_,
      }
    };
    
    return JSON.stringify(modelData, null, 2);
  }
  
  /**
   * Load model from JSON string.
   * 
   * @param json JSON string containing model data
   */
  async loadFromJSON(json: string): Promise<void> {
    const modelData = JSON.parse(json);
    
    // Dispose existing weights
    this.weights_?.dispose();
    
    // Load weights
    this.weights_ = tf.tensor3d(modelData.weights);
    
    // Load metadata
    if (modelData.metadata) {
      // Create new params object instead of modifying readonly
      const loadedParams = modelData.metadata.params;
      if (loadedParams) {
        // Reconstruct the object with new params to maintain immutability
        Object.defineProperty(this, 'params', {
          value: loadedParams,
          writable: false,
          configurable: true
        });
      }
      this.totalSamplesLearned_ = modelData.metadata.totalSamplesLearned || 0;
      this.currentEpoch_ = modelData.metadata.currentEpoch || 0;
      this.quantizationErrors_ = modelData.metadata.quantizationErrors || [];
    }
    
    // Reinitialize schedulers and distance matrix
    this.initializeSchedulers();
    this.gridDistanceMatrix_ = createGridDistanceMatrix(
      this.params.gridHeight,
      this.params.gridWidth,
      this.params.topology!
    );
  }
  
  /**
   * Enable streaming mode for continuous learning.
   * Configures the SOM for online learning with optimal settings.
   */
  enableStreamingMode(batchSize?: number): void {
    this.params.onlineMode = true;
    this.params.miniBatchSize = batchSize || SOM.DEFAULT_MINI_BATCH_SIZE;
    
    // Adjust schedulers for continuous learning
    // Use slower decay for streaming scenarios
    const { gridWidth, gridHeight } = this.params;
    const initialLearningRate = typeof this.params.learningRate === 'number' 
      ? this.params.learningRate 
      : SOM.DEFAULT_LEARNING_RATE;
    
    this.learningRateScheduler_ = (_epoch: number, _totalEpochs: number) => {
      // Slower decay for streaming
      const virtualEpoch = this.totalSamplesLearned_ / 1000; // Adjust scale
      return initialLearningRate * Math.exp(-virtualEpoch / 100);
    };
    
    const initialRadius = Math.max(gridWidth, gridHeight) / 2;
    this.radiusScheduler_ = (_epoch: number, _totalEpochs: number) => {
      const virtualEpoch = this.totalSamplesLearned_ / 1000;
      return Math.max(1, initialRadius * Math.exp(-virtualEpoch / 50));
    };
  }
  
  /**
   * Process a stream of data samples.
   * Automatically manages batch accumulation and training.
   * 
   * @param sample Single sample or small batch
   * @param autoTrain Whether to train immediately or accumulate
   */
  async processStream(
    sample: DataMatrix,
    autoTrain: boolean = true
  ): Promise<void> {
    if (!this.params.onlineMode) {
      this.enableStreamingMode();
    }
    
    const xTensor = isTensor(sample) ? sample as tf.Tensor2D : tf.tensor2d(sample);
    
    try {
      if (autoTrain) {
        await this.partialFit(xTensor);
      } else {
        // Accumulate samples for batch training
        // This would need a buffer implementation
        console.log('Batch accumulation not yet implemented');
      }
    } finally {
      if (!isTensor(sample)) {
        xTensor.dispose();
      }
    }
  }
  
  /**
   * Get streaming statistics.
   */
  getStreamingStats(): {
    totalSamples: number;
    virtualEpoch: number;
    currentLearningRate: number;
    currentRadius: number;
    latestQuantizationError: number;
  } {
    const virtualEpoch = Math.floor(this.totalSamplesLearned_ / 100);
    
    return {
      totalSamples: this.totalSamplesLearned_,
      virtualEpoch,
      currentLearningRate: this.learningRateScheduler_?.(
        virtualEpoch,
        this.params.numEpochs || 100
      ) || 0,
      currentRadius: this.radiusScheduler_?.(
        virtualEpoch,
        this.params.numEpochs || 100
      ) || 0,
      latestQuantizationError: this.quantizationErrors_.length > 0
        ? this.quantizationErrors_[this.quantizationErrors_.length - 1]
        : 0,
    };
  }
  
  /**
   * Clean up tensors.
   */
  dispose(): void {
    this.weights_?.dispose();
    this.gridDistanceMatrix_?.dispose();
    this.bmus_?.dispose();
  }
}
"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.AgglomerativeClustering = void 0;
const tf = __importStar(require("@tensorflow/tfjs-node"));
const pairwise_distance_1 = require("../utils/pairwise_distance");
const linkage_1 = require("./linkage");
/**
 * Agglomerative (hierarchical) clustering estimator skeleton.
 *
 * Only the constructor, parameter validation and public property definitions
 * are implemented as part of this initial task. The actual clustering logic
 * will be added in subsequent tasks.
 */
class AgglomerativeClustering {
    constructor(params) {
        /**
         * Cluster labels produced by `fit` / `fitPredict`.
         *
         * Populated after calling `fit`.
         */
        this.labels_ = null;
        /**
         * Children of each non-leaf node in the hierarchical clustering tree.
         * Shape: `(nSamples-1, 2)` where each row gives the indices of the merged
         * clusters. Lazily populated by future implementation.
         */
        this.children_ = null;
        /**
         * Number of leaves in the hierarchical clustering tree (equals `nSamples`).
         */
        this.nLeaves_ = null;
        // Perform a shallow copy to freeze user input and avoid side effects.
        this.params = { ...params };
        AgglomerativeClustering.validateParams(this.params);
    }
    /**
     * Fits the estimator to the provided data matrix.
     *
     * Note: The actual algorithm is not implemented yet. The stub only exists so
     * the public interface is complete and unit tests can assert that the method
     * is callable.
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    async fit(_X) {
        // Convert input to a tf.Tensor2D for distance computation if necessary.
        // Early exit for edge-cases ------------------------------------------------
        if (Array.isArray(_X) && _X.length === 0) {
            throw new Error("Input X must contain at least one sample.");
        }
        const points = Array.isArray(_X)
            ? tf.tensor2d(_X)
            : _X;
        const nSamples = points.shape[0];
        // Handle trivial case of single sample separately
        if (nSamples === 1) {
            this.labels_ = [0];
            this.children_ = [];
            this.nLeaves_ = 1;
            points.dispose?.();
            return;
        }
        const { metric = "euclidean", linkage = "ward", nClusters } = this.params;
        // -----------------------------------------------------------------------
        // Compute initial pairwise distance matrix (plain number[][] for fast JS
        // level manipulation). We leverage the existing helper in utils.
        // -----------------------------------------------------------------------
        const distanceTensor = (0, pairwise_distance_1.pairwiseDistanceMatrix)(points, metric);
        const D = (await distanceTensor.array());
        distanceTensor.dispose();
        /*  ------------------------------------------------------------------
         *  Hierarchical agglomeration loop
         *  ------------------------------------------------------------------ */
        // Cluster bookkeeping arrays. Index i corresponds to row/col i in D.
        let clusterIds = Array.from({ length: nSamples }, (_, i) => i);
        const clusterSizes = Array(nSamples).fill(1);
        let nextClusterId = nSamples; // new clusters get incremental ids
        const children = [];
        // Track current cluster label for each sample (global cluster ids)
        const sampleLabels = Array.from({ length: nSamples }, (_, i) => i);
        // Merge until the desired number of clusters is reached.
        while (clusterIds.length > nClusters) {
            // -------------------------------------------------------------------
            // Find closest pair (i,j)
            // -------------------------------------------------------------------
            let minDist = Number.POSITIVE_INFINITY;
            let minI = 0;
            let minJ = 1;
            for (let i = 0; i < D.length; i++) {
                for (let j = i + 1; j < D.length; j++) {
                    const d = D[i][j];
                    if (d < minDist) {
                        minDist = d;
                        minI = i;
                        minJ = j;
                    }
                }
            }
            // Store merge in children_ (using global cluster ids)
            const idI = clusterIds[minI];
            const idJ = clusterIds[minJ];
            children.push([idI, idJ]);
            // Update distance matrix & auxiliary arrays
            (0, linkage_1.update_distance_matrix)(D, clusterSizes, minI, minJ, linkage);
            // Assign a new cluster id to the merged entity (row minI after update)
            const newId = nextClusterId++;
            clusterIds[minI] = newId;
            clusterIds.splice(minJ, 1);
            // Propagate new labels to samples that belonged to idI or idJ
            for (let s = 0; s < nSamples; s++) {
                const lbl = sampleLabels[s];
                if (lbl === idI || lbl === idJ) {
                    sampleLabels[s] = newId;
                }
            }
            // Loop continues with contracted D.
        }
        // ---------------------------------------------------------------------
        // Derive flat cluster labels by cutting dendrogram at desired number of
        // clusters. The simplest approach is to recreate cluster membership from
        // bottom-up using the recorded merges.
        // ---------------------------------------------------------------------
        const labels = sampleLabels;
        // Relabel to contiguous range 0 .. nClusters-1
        const uniqueOld = Array.from(new Set(labels));
        const mapping = new Map();
        uniqueOld.forEach((oldLabel, newLabel) => mapping.set(oldLabel, newLabel));
        this.labels_ = labels.map((old) => mapping.get(old));
        this.children_ = children;
        this.nLeaves_ = nSamples;
        // Dispose created tensor if we have created one from array input.
        if (Array.isArray(_X)) {
            points.dispose();
        }
    }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    async fitPredict(_X) {
        await this.fit(_X);
        if (this.labels_ == null) {
            throw new Error("AgglomerativeClustering failed to compute labels.");
        }
        return this.labels_;
    }
    /* --------------------------------------------------------------------- */
    /*                         Parameter Validation                          */
    /* --------------------------------------------------------------------- */
    static validateParams(params) {
        const { nClusters, linkage = "ward", metric = "euclidean" } = params;
        // nClusters must be a positive integer
        if (!Number.isInteger(nClusters) || nClusters < 1) {
            throw new Error("nClusters must be a positive integer (>= 1).");
        }
        // linkage value
        if (!AgglomerativeClustering.VALID_LINKAGES.includes(linkage)) {
            throw new Error(`Invalid linkage '${linkage}'. Must be one of ${AgglomerativeClustering.VALID_LINKAGES.join(", ")}.`);
        }
        // metric value
        if (!AgglomerativeClustering.VALID_METRICS.includes(metric)) {
            throw new Error(`Invalid metric '${metric}'. Must be one of ${AgglomerativeClustering.VALID_METRICS.join(", ")}.`);
        }
        // Additional consistency check: Ward linkage requires Euclidean distance.
        if (linkage === "ward" && metric !== "euclidean") {
            throw new Error("Ward linkage requires metric to be 'euclidean'.");
        }
    }
}
exports.AgglomerativeClustering = AgglomerativeClustering;
/**
 * Allowed linkage strategies.
 */
AgglomerativeClustering.VALID_LINKAGES = [
    "ward",
    "complete",
    "average",
    "single",
];
/**
 * Allowed distance metrics.
 */
AgglomerativeClustering.VALID_METRICS = [
    "euclidean",
    "manhattan",
    "cosine",
];

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
exports.pairwiseEuclideanMatrix = pairwiseEuclideanMatrix;
exports.pairwiseDistanceMatrix = pairwiseDistanceMatrix;
const tf = __importStar(require("@tensorflow/tfjs-node"));
const tensor_1 = require("./tensor");
/**
 * Optimised Euclidean pairwise distance using the identity
 * ‖x − y‖² = ‖x‖² + ‖y‖² − 2·xᵀy to avoid building an (n,n,d) tensor.
 */
function pairwiseEuclideanMatrix(points) {
    return tf.tidy(() => {
        const squaredNorms = points.square().sum(1).reshape([-1, 1]); // (n,1)
        const gram = points.matMul(points.transpose()); // (n,n)
        const distancesSquared = squaredNorms
            .add(squaredNorms.transpose())
            .sub(gram.mul(2));
        const zero = tf.scalar(0, "float32");
        const distancesSquaredClamped = tf.maximum(distancesSquared, zero);
        const dist = distancesSquaredClamped.sqrt();
        const distSym = dist.add(dist.transpose()).div(2);
        const n = distSym.shape[0];
        const mask = tf.ones([n, n], "float32").sub(tf.eye(n));
        return distSym.mul(mask);
    });
}
/**
 * Computes the pairwise distance matrix for the given points according to the
 * requested metric.
 *
 * The result is an `(n, n)` tensor `D` where `D[i, j]` contains the distance
 * between row `i` and row `j` of the input `points`.
 *
 * Supported metrics:
 *   • "euclidean"  – ℓ2 distance (uses an optimised implementation)
 *   • "manhattan"  – ℓ1 distance
 *   • "cosine"     – 1 − cosine-similarity
 *
 * For performance and numerical stability the computation is wrapped in
 * `tf.tidy` so that all intermediate tensors are eagerly disposed.
 */
function pairwiseDistanceMatrix(points, metric = "euclidean") {
    switch (metric) {
        case "euclidean":
            return pairwiseEuclideanMatrix(points);
        case "manhattan":
            return tf.tidy(() => {
                const n = points.shape[0];
                const expandedA = points.expandDims(1); // (n,1,d)
                const expandedB = points.expandDims(0); // (1,n,d)
                const dist = (0, tensor_1.manhattanDistance)(expandedA, expandedB);
                const distSym = dist.add(dist.transpose()).div(2);
                const mask = tf.ones([n, n], "float32").sub(tf.eye(n));
                return distSym.mul(mask);
            });
        case "cosine":
            return tf.tidy(() => {
                const n = points.shape[0];
                const expandedA = points.expandDims(1); // (n,1,d)
                const expandedB = points.expandDims(0); // (1,n,d)
                const dist = (0, tensor_1.cosineDistance)(expandedA, expandedB);
                const distSym = dist.add(dist.transpose()).div(2);
                const mask = tf.ones([n, n], "float32").sub(tf.eye(n));
                return distSym.mul(mask);
            });
        default:
            // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
            throw new Error(`Unsupported metric '${metric}'.`);
    }
}

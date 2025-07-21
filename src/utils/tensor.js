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
exports.pairwiseEuclideanMatrix = void 0;
exports.arrayToTensor = arrayToTensor;
exports.tensorToArray = tensorToArray;
exports.euclideanDistance = euclideanDistance;
exports.manhattanDistance = manhattanDistance;
exports.cosineDistance = cosineDistance;
const tf = __importStar(require("@tensorflow/tfjs-node"));
/**
 * Converts a regular (nested) JavaScript array into a TensorFlow.js tensor
 * with the provided dtype (defaults to `float32`).
 *
 * The function is wrapped in `tf.tidy` to ensure that any intermediate
 * tensors that may be created by TensorFlow.js during conversion are
 * automatically disposed of.
 */
function arrayToTensor(arr, dtype = "float32") {
    return tf.tidy(() => tf.tensor(arr, undefined, dtype));
}
/**
 * Converts a tensor back to a JavaScript array (synchronously).
 *
 * The returned value is a *copy* of the underlying data, so further
 * manipulations will not affect the original tensor.
 */
function tensorToArray(tensor) {
    // Using .arraySync() is safe here because callers explicitly request the
    // data as a JS structure. For large tensors prefer the async variant.
    return tensor.arraySync();
}
/* ------------------------------------------------------------------------- */
/*                        Distance / Similarity Metrics                      */
/* ------------------------------------------------------------------------- */
/**
 * Computes the element-wise Euclidean (ℓ2) distance between two tensors along
 * their last dimension.
 *
 * Both inputs must be broadcast-compatible. The result will have the broadcast
 * shape of `tf.sub(a, b).sum(-1)` (i.e. the shapes minus the last dimension).
 *
 * Example:
 * ```ts
 * const a = tf.tensor([[0, 0], [1, 1]]); // (2, 2)
 * const b = tf.tensor([1, 0]);           // (2)
 * euclideanDistance(a, b)  // => Tensor([1, 1])
 * ```
 */
function euclideanDistance(a, b) {
    return tf.tidy(() => tf.sub(a, b).square().sum(-1).sqrt());
}
/**
 * Computes the Manhattan (ℓ1) distance between two tensors along their last
 * dimension.
 */
function manhattanDistance(a, b) {
    return tf.tidy(() => a.sub(b).abs().sum(-1));
}
/**
 * Computes the cosine distance (1 ‑ cosine similarity) between two tensors
 * along their last dimension.
 */
function cosineDistance(a, b) {
    return tf.tidy(() => {
        const aNorm = a.norm();
        const bNorm = b.norm();
        const dot = a.mul(b).sum(-1);
        const eps = tf.scalar(1e-8);
        const denom = aNorm.mul(bNorm).add(eps);
        const similarity = dot.div(denom);
        return tf.scalar(1).sub(similarity);
    });
}
/* ------------------------------------------------------------------------- */
/*                            Broadcast Utilities                            */
/* ------------------------------------------------------------------------- */
/**
 * Efficiently computes pairwise Euclidean distance matrix for a set of points
 * represented by a 2-D tensor of shape `(n, d)`.
 *
 * The implementation uses the well-known trick
 * ‖x − y‖² = ‖x‖² + ‖y‖² − 2·xᵀy with broadcasting to avoid allocating an
 * `(n, n, d)` intermediate tensor.
 */
// Re-export to maintain backward compatibility while delegating to the new
// implementation in `pairwise_distance.ts`.
var pairwise_distance_1 = require("./pairwise_distance");
Object.defineProperty(exports, "pairwiseEuclideanMatrix", { enumerable: true, get: function () { return pairwise_distance_1.pairwiseEuclideanMatrix; } });

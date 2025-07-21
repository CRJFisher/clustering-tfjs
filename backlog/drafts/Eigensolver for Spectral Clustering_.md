# A Pragmatic Guide to High-Accuracy Eigensolvers in JavaScript for Spectral Clustering

## **Section 1: Introduction and Problem Framing**

The successful application of many advanced machine learning algorithms hinges on the accuracy and stability of their underlying numerical components. When these components fail, even subtly, the entire model can produce erroneous results. This report addresses such a scenario within a spectral clustering implementation, where an existing Jacobi-based eigensolver provides insufficient accuracy, leading to downstream failures. The objective of this analysis is to diagnose the root cause of this issue, survey superior alternative algorithms, and provide a definitive, pragmatic, and implementable solution using modern JavaScript frameworks.

### **1.1 The Role of Eigendecomposition in Spectral Clustering**

Spectral clustering is a powerful technique that can identify non-convex cluster structures where distance-based algorithms like k-means might fail. Its efficacy stems from its re-framing of the clustering problem from a geometric one to a graph-based one. Data points are treated as nodes in a graph, and the connections (edges) between them are weighted by a measure of similarity, or affinity. A common choice for this is the Radial Basis Function (RBF) kernel, which produces a symmetric affinity matrix A.  
From this affinity matrix, a graph Laplacian matrix is constructed. The unnormalized graph Laplacian is defined as L=D−A, where D is the diagonal degree matrix with Dii​=∑j​Aij​. This Laplacian matrix is central to spectral methods and possesses several critical properties: it is symmetric and positive semi-definite.2 These properties are fundamental because they guarantee that the eigenvalues of  
L are real and non-negative, and that its eigenvectors form an orthogonal basis.  
The core insight of spectral clustering is that the eigenvectors corresponding to the smallest non-zero eigenvalues of the Laplacian provide a new, low-dimensional embedding of the data.4 In this new "spectral" space, the clusters that were originally intertwined and non-convex become linearly separable. The final step is to apply a simpler clustering algorithm, typically k-means, to the rows of the matrix formed by these eigenvectors. This process effectively transforms a difficult clustering problem into a simple one. Consequently, the accuracy of the entire spectral clustering pipeline is critically dependent on the accurate computation of these specific eigenvectors. An inaccurate eigensolver will produce a flawed embedding, leading directly to incorrect cluster assignments by k-means.

### **1.2 Diagnosing the Failure: The Limits of the Jacobi Method**

The current implementation utilizes the Jacobi eigenvalue algorithm, a classic method that computes the full eigendecomposition of a symmetric matrix by applying a sequence of plane rotations (Givens rotations) to zero out off-diagonal elements iteratively.4 While conceptually straightforward and historically important, the Jacobi method can exhibit limitations in numerical precision, especially when high accuracy is demanded for specific eigenvectors of ill-conditioned matrices or when eigenvalues are closely clustered.5  
The context of the failing RBF tests reveals the severity of this limitation. An observed eigenvector difference of approximately 0.0065 per component, while numerically small, is semantically catastrophic. In the spectral embedding, the sign and relative magnitude of each component of an eigenvector determine a data point's position along a particular axis. A small numerical error can be sufficient to "flip" a point from one side of a k-means decision boundary to the other. When this happens to multiple points near cluster boundaries, the resulting cluster assignments change, causing a significant drop in the Adjusted Rand Index (ARI), a measure of clustering similarity. The reported ARI values of 0.77-0.93, falling short of the required 0.95, are a direct symptom of this underlying numerical imprecision.  
The investigation has already confirmed that the Jacobi solver has reached its practical limits. Attempts to improve accuracy by tightening tolerances and increasing iteration counts have yielded no improvement. This indicates that the issue is not one of insufficient iteration but a fundamental limitation of the algorithm's ability to converge to the true eigenvectors with the required fidelity in finite-precision arithmetic for these specific test cases. The problem, therefore, is not in the k-means post-processing but lies squarely within the eigenvector computation step.  
A crucial factor shaping the solution space is the implementation environment: TypeScript/JavaScript running in Node.js, with a dependency on tensorflow.js. The JavaScript numerical ecosystem, while rapidly maturing, does not possess the decades of refinement found in Fortran or C++ libraries like LAPACK and ARPACK, which are the gold standard in scientific computing.7 Many pure JavaScript libraries have known limitations; for example, the popular  
numeric.js library is documented to throw internal errors and fail on certain types of matrices, making it an unreliable choice for production systems.8 This reality reframes the problem: the search is not merely for a theoretically superior algorithm, but for a superior algorithm whose core computational primitives are robustly and reliably implemented in a readily available JavaScript library. The investigation must therefore prioritize algorithms that can be constructed from fundamental, well-supported operations like matrix-vector multiplication and, critically, the solving of linear systems.

### **1.3 Defining the Path Forward: Criteria for a Successful Solution**

Based on the diagnosis, a successful replacement for the Jacobi eigensolver must satisfy a clear set of criteria derived from the project's acceptance goals. The proposed solution will be evaluated against these four pillars:

1. **Exceed Accuracy Thresholds:** The primary goal is to achieve sufficient numerical precision in the computed eigenvectors to pass all failing RBF kernel tests. This translates to a target ARI of ≥0.95, which requires eigenvector components to be accurate enough to prevent incorrect cluster assignments.
2. **Be Pragmatically Implementable:** The algorithm must be expressible in pure TypeScript/JavaScript. It cannot rely on native C++/Fortran bindings or require a full port of a complex library like ARPACK. The implementation should leverage the capabilities of a robust numerical library available in the Node.js ecosystem, with tensorflow.js being the preferred choice given its existing use in the project.
3. **Target the Smallest Eigenvalues:** Spectral clustering requires the eigenvectors corresponding to the k smallest non-zero eigenvalues of the graph Laplacian. The chosen algorithm should be efficient at finding a small subset of eigenvalues at one end of the spectrum, rather than computing the entire spectrum, which is wasteful.4
4. **Prioritize Accuracy over Speed:** While computational performance is a long-term concern, the immediate priority is correctness and accuracy. An algorithm that is slower but provides the necessary precision to pass the tests is preferable to a faster but less accurate one.

The following sections will survey candidate algorithms through the lens of these criteria, leading to a strategic recommendation and a detailed implementation blueprint designed to resolve the current accuracy bottleneck.

## **Section 2: Survey of Candidate Iterative Eigensolvers**

To find a suitable replacement for the Jacobi method, the focus shifts to iterative eigensolvers. Unlike direct methods that transform the entire matrix, iterative methods generate a sequence of approximate solutions that converge to the desired eigenpairs. They are particularly well-suited for large, often sparse, matrices where only a small fraction of the eigenspectrum is required, which is precisely the case in spectral clustering.4

### **2.1 The Foundational Power Iteration Method**

The Power Iteration, or Power Method, is the conceptual cornerstone of many advanced iterative eigensolvers.4 Its mechanism is remarkably simple: starting with an initial random vector  
b0​, it iteratively computes the sequence:  
bk+1​=∥Abk​∥Abk​​  
where A is the matrix of interest.10 If the matrix  
A has a dominant eigenvalue λ1​ (i.e., an eigenvalue that is strictly greater in absolute value than all other eigenvalues, ∣λ1​∣\>∣λ2​∣≥⋯≥∣λn​∣), and the initial vector b0​ has a non-zero component in the direction of the corresponding eigenvector v1​, the sequence of vectors bk​ will converge to v1​.10  
The convergence of the power method is linear, with the error at each step being reduced by a factor of ∣λ2​/λ1​∣.10 This means convergence can be very slow if the dominant and second-dominant eigenvalues are close in magnitude. While the power method is simple and effective for finding the largest eigenvalue (and is famously used in algorithms like Google's PageRank), it is not directly applicable to the problem at hand, which requires the  
_smallest_ eigenvalues of the graph Laplacian.10 However, its core idea—iterative matrix-vector multiplication and normalization—forms the basis for the more sophisticated methods that follow.

### **2.2 Inverse Iteration: Targeting the Smallest Eigenvalues**

To adapt the power method for finding the smallest eigenvalues, one can leverage a simple but powerful property of matrix inverses. If a matrix A has eigenvalues λi​, its inverse A−1 has eigenvalues 1/λi​ with the same corresponding eigenvectors.12 Therefore, the largest magnitude eigenvalue of  
A−1 corresponds to the smallest magnitude eigenvalue of A. Applying the power method to A−1 yields the **Inverse Iteration** algorithm:  
bk+1​=∥A−1bk​∥A−1bk​​  
This process converges to the eigenvector associated with the smallest eigenvalue of A.12  
This concept can be generalized into the **Shift-and-Invert** strategy. For any scalar shift σ that is not an eigenvalue of A, the matrix (A−σI)−1 has eigenvalues 1/(λi​−σ).15 Applying the power method to this shifted-and-inverted matrix will cause it to converge to the eigenvector whose corresponding eigenvalue  
λi​ is closest to the shift σ.17 This is an extremely powerful technique for targeting specific interior eigenvalues. For the spectral clustering problem, the goal is to find the eigenvalues closest to zero, so a shift of  
σ=0 is used, reducing the method to the basic inverse iteration.  
A critical implementation detail is that one should **never** explicitly compute the matrix inverse (A−σI)−1. Inverting a matrix is computationally expensive (typically an O(n3) operation) and numerically unstable.12 Instead, the iterative step is reformulated as solving a system of linear equations 16:  
Solve (A−σI)yk+1​=bk​ for yk+1​  
bk+1​=∥yk+1​∥yk+1​​  
This linear system solve is the computational core of the algorithm. While still a significant operation, efficient and stable methods like LU decomposition can be used, and the factorization needs to be computed only once if the shift σ is fixed.12 The convergence rate of inverse iteration is linear, but the convergence factor depends on the ratio of the eigenvalues of the  
_inverted_ matrix. This means that by choosing a shift σ very close to the target eigenvalue, the convergence can be made exceptionally fast.17

### **2.3 Rayleigh Quotient Iteration (RQI): The High-Speed, High-Accuracy Contender**

Rayleigh Quotient Iteration (RQI) is a sophisticated enhancement of the inverse iteration method that dramatically accelerates convergence.20 Instead of using a fixed, pre-determined shift  
σ, RQI dynamically computes the optimal shift at each iteration. This optimal shift is the **Rayleigh Quotient**, defined for a vector vk​ as:  
ρk​=vkT​vk​vkT​Avk​​  
The Rayleigh quotient provides the best possible estimate of an eigenvalue for a given approximate eigenvector vk​.11 By using this dynamically updated, highly accurate shift, RQI converges with remarkable speed.  
The RQI algorithm proceeds as follows 21:

1. Start with a normalized initial vector v0​.
2. For k=0,1,2,… until convergence:  
   a. Compute the Rayleigh quotient as the current shift: ρk​=vkT​Avk​.  
   b. Solve the linear system for the next vector approximation: (A−ρk​I)yk+1​=vk​.  
   c. Normalize the result: vk+1​=yk+1​/∥yk+1​∥.

The key advantage of RQI lies in its convergence rate. For a general matrix, convergence is quadratic. However, for symmetric (or Hermitian) matrices—such as the graph Laplacian—RQI exhibits **cubic convergence**.20 This means that the number of correct significant digits in the eigenvalue approximation roughly triples with each iteration. This rate of convergence is so fast that it has been described as "mind-boggling" 19, and in practice, only a handful of iterations are needed to achieve very high precision.11 This makes RQI an exceptionally strong candidate for applications demanding high accuracy.

### **2.4 The Lanczos Algorithm: The Method Behind ARPACK**

The Lanczos algorithm is a powerful iterative method that forms the foundation of many state-of-the-art eigensolvers, including the ARPACK library used by scipy.sparse.linalg.eigsh in Python.7 The algorithm is an adaptation of the power method designed specifically for large, sparse, symmetric matrices.9  
Its core mechanism involves building an orthonormal basis for a **Krylov subspace**. The Krylov subspace of dimension m generated by a matrix A and a starting vector v is the space spanned by the vectors {v,Av,A2v,…,Am−1v}.9 The Lanczos algorithm iteratively generates a set of orthonormal vectors  
q1​,q2​,…,qm​ that span this subspace. In this orthonormal basis, the projection of the original matrix A takes the form of a much smaller m×m symmetric tridiagonal matrix, T.22 The eigenvalues of this small tridiagonal matrix, known as Ritz values, serve as excellent approximations to the extremal (largest and smallest) eigenvalues of the original large matrix  
A.26  
However, a from-scratch implementation of the Lanczos algorithm faces a significant and well-known challenge: **numerical instability**. In finite-precision arithmetic, the theoretically orthogonal Lanczos vectors (qi​) quickly lose their mutual orthogonality due to round-off errors.22 This loss of orthogonality is a critical failure, as it leads to the appearance of spurious or "ghost" eigenvalues and degrades the accuracy of the computed eigenpairs.28  
To combat this instability, practical implementations must incorporate a **reorthogonalization** strategy.24

- **Full Reorthogonalization:** At each step, the newly generated vector is explicitly made orthogonal to _all_ previously generated vectors using a process like the Gram-Schmidt procedure. This is the most robust approach but is computationally expensive, as the cost per iteration grows with the number of iterations.22
- **Selective or Partial Reorthogonalization:** These are more complex schemes that monitor the loss of orthogonality and perform reorthogonalization only when necessary, attempting to balance robustness with computational cost.27

Implementing any of these reorthogonalization schemes correctly is a non-trivial task, significantly increasing the complexity of a pure JavaScript Lanczos solver. Furthermore, just like inverse iteration, the standard Lanczos algorithm is best at finding extremal eigenvalues. To find the smallest eigenvalues (which are near the interior of the spectrum if the matrix has both positive and negative eigenvalues) or to accelerate convergence to the smallest positive eigenvalues, Lanczos must be paired with a **shift-and-invert** strategy.5 This means applying the Lanczos algorithm to the operator  
(A−σI)−1, which once again brings back the requirement of solving a linear system at every single iteration.  
The analysis of these candidate methods reveals a unifying theme. The most powerful and relevant algorithms for this task—Inverse Iteration, Rayleigh Quotient Iteration, and a production-grade Shift-and-Invert Lanczos—all depend on the same core computational primitive: the solution of a linear system of the form (A−μI)y=x. This realization is pivotal. It shifts the basis of comparison away from the raw complexity of the underlying matrix operations and towards the elegance, efficiency, and convergence properties of the "scaffolding" built around this common linear solve. The question becomes: which algorithm provides the best accuracy and convergence for the least amount of implementation complexity in that scaffolding?

## **Section 3: Comparative Analysis and Strategic Recommendation**

With a theoretical understanding of the candidate algorithms, the next step is to evaluate them systematically against the project's specific criteria. This analysis will illuminate the trade-offs between accuracy, implementation complexity, and suitability for the task, leading to a clear and defensible recommendation for the most pragmatic path forward.

### **3.1 Evaluating Candidates Against Project Criteria**

A methodical evaluation reveals a distinct front-runner that optimally balances all requirements.

- **Jacobi Method (Baseline):**
  - **Accuracy:** Fails. As established, the Jacobi method is the source of the problem, providing insufficient precision for the RBF kernel test cases.
  - **Implementation Feasibility:** High (already implemented).
  - **Suitability:** Poor. While it finds all eigenvalues, its accuracy limitations make it unsuitable for this high-precision application.
- **Lanczos Algorithm (with Full Reorthogonalization):**
  - **Accuracy:** High. A correctly implemented Lanczos algorithm with shift-and-invert is the basis for ARPACK and would certainly meet the accuracy requirements.
  - **Implementation Feasibility:** Very Low. This is the "Lanczos Trap." While theoretically ideal, a robust, from-scratch implementation in JavaScript is a formidable challenge. It requires not only a linear solver for the shift-and-invert strategy but also a carefully managed full reorthogonalization procedure to maintain numerical stability.22 The complexity of managing the Krylov basis, the tridiagonal matrix, and the reorthogonalization logic far exceeds that of the other iterative methods, making it an impractical choice for this project's constraints.27
  - **Suitability:** High in theory, but pragmatically poor due to implementation complexity.
- **Inverse Iteration:**
  - **Accuracy:** High. By repeatedly solving a linear system, it refines the eigenvector to a high degree of precision, limited primarily by machine epsilon and the solver's quality. It is a significant step up from the Jacobi method.
  - **Implementation Feasibility:** Moderate. The algorithm's logic is a simple loop built around a linear solver, a primitive readily available in libraries like tensorflow.js.32
  - **Suitability:** Good. It directly targets the smallest eigenvalue (with a shift of σ=0) and is straightforward to implement. Its primary drawback is its linear convergence rate, which might require more iterations than more advanced methods.11
- **Rayleigh Quotient Iteration (RQI):**
  - **Accuracy:** Very High. The cubic convergence rate for symmetric matrices ensures that RQI reaches extremely high precision in a minimal number of iterations.20 This is its standout feature and directly addresses the core problem.
  - **Implementation Feasibility:** Moderate. The implementation complexity of RQI is virtually identical to that of Inverse Iteration. It is also a simple loop centered around a linear solve, with the only addition being the calculation of the Rayleigh quotient at each step—a simple sequence of matrix-vector and vector-vector products.19
  - **Suitability:** Excellent. It combines the implementation simplicity of Inverse Iteration with a vastly superior convergence rate, making it the most efficient and effective choice for achieving the required accuracy.

### **3.2 The Pragmatic Sweet Spot: Why RQI is the Optimal Choice**

The comparative analysis unequivocally points to **Rayleigh Quotient Iteration (RQI)** as the optimal solution. It occupies a pragmatic sweet spot, delivering the highest possible performance on the most critical metrics without introducing prohibitive complexity.  
The decision rests on the relationship between implementation effort and the resulting accuracy and convergence. Both Inverse Iteration and RQI are built around the same core operation: solving a linear system. The effort to implement the loop structure for either is comparable. However, the return on that investment is vastly different. Inverse Iteration offers linear convergence, which is reliable but can be slow. RQI, for the same implementation effort, delivers cubic convergence.21 This is not an incremental improvement; it is a fundamental change in the algorithm's behavior, allowing it to achieve the target accuracy in a fraction of the iterations required by other methods.11  
In contrast, the Lanczos algorithm, while the theoretical parent of the incumbent ARPACK solver, presents a poor effort-to-reward ratio for a from-scratch implementation. The significant added complexity of managing the Krylov basis and ensuring its orthogonality through reorthogonalization offers no practical advantage over the much simpler RQI for finding a few eigenpairs of a dense symmetric matrix, especially when RQI's convergence is already exceptionally fast.  
Finally, RQI aligns perfectly with the existing technology stack. The tensorflow.js library provides the tf.linalg.solve function, which is the exact primitive needed to implement the core of RQI.32 This avoids the need to introduce new, potentially less reliable dependencies and allows the solution to be built on a robust, well-maintained foundation.

### **3.3 Table: Comparative Analysis of Eigensolver Algorithms**

The following table summarizes the evaluation of the candidate algorithms, highlighting the clear advantages of Rayleigh Quotient Iteration for this specific task.

| Feature                       | Jacobi                  | Lanczos (w/ Full Re-ortho)           | Inverse Iteration          | Rayleigh Quotient Iteration (RQI)    |
| :---------------------------- | :---------------------- | :----------------------------------- | :------------------------- | :----------------------------------- |
| **Primary Mechanism**         | Matrix Rotations        | Krylov Subspace / Tridiagonalization | Power Method on A−1        | Inverse Iteration w/ Dynamic Shift   |
| **Accuracy**                  | Moderate (Insufficient) | High                                 | High                       | **Very High**                        |
| **Convergence Rate**          | N/A (Direct-like)       | Linear (for Ritz values)             | Linear                     | **Cubic** (for symmetric)            |
| **Implementation Complexity** | High                    | **Very High**                        | Moderate                   | Moderate                             |
| **Core Operation**            | Rotations               | Mat-Vec, Orthogonalization           | Linear Solve               | Linear Solve                         |
| **Finds**                     | All Eigenvalues         | Extremal Eigenvalues                 | Eigenvalue nearest shift σ | Eigenvalue nearest shift σ           |
| **Suitability for Task**      | **Poor**                | **High (but too complex)**           | **Good**                   | **Excellent (Pragmatic Sweet Spot)** |

## **Section 4: A TypeScript Implementation Blueprint for RQI**

This section provides a detailed, practical guide to implementing the recommended Rayleigh Quotient Iteration (RQI) algorithm in TypeScript. It covers the choice of numerical libraries, a step-by-step implementation for a single eigenpair, and a strategy for extending the method to find the k smallest eigenvectors required for spectral clustering.

### **4.1 Choosing the Right Tools: The JavaScript Linear Algebra Landscape**

The success of any numerical algorithm implementation depends heavily on the quality of its foundational library. In the JavaScript ecosystem, several options exist, but they vary significantly in robustness, features, and performance.

- **tensorflow.js (Primary Recommendation):** This is the ideal choice for this project. It is already part of the technology stack, is actively maintained by Google, and offers the potential for GPU acceleration for matrix operations.34 Most importantly, it provides a robust implementation of the critical primitive needed for RQI:  
  tf.linalg.solve.32 While it does not have a high-level "eigensolver" function, its powerful low-level linear algebra capabilities are perfectly suited for building one.36
- **ml-matrix (Recommended Fallback and Validation Tool):** The ml-matrix library is a strong alternative and an excellent tool for validation. It is a pure JavaScript, CPU-based library that has been praised for its correctness and stability, succeeding where other libraries like numeric.js have failed.8 It offers both a linear solver (  
  solve, which can use SVD for singular matrices) and a full EigenvalueDecomposition class.37 This makes it invaluable for two purposes: 1\) as a fallback implementation if  
  tensorflow.js cannot be used in a particular environment, and 2\) as a "ground truth" reference to validate the correctness of the custom RQI implementation during development.
- **Libraries to Avoid:**
  - numeric.js: Despite its popularity, there are multiple documented reports of it throwing internal errors and failing on specific matrix types, particularly sparse or singular ones.8 This unreliability makes it unsuitable for a production-grade component.
  - math.js: The documentation for its eigs function explicitly states that it uses "traditional" methods and is "not a modern, high-precision eigenvalue computation".38 This directly contradicts the primary goal of improving accuracy.

The following table justifies the tooling choices based on the features essential for this task.

| Library           | solve(A, b)                           | eigs() / EigenvalueDecomposition | Key Considerations                                                            | Recommendation                          |
| :---------------- | :------------------------------------ | :------------------------------- | :---------------------------------------------------------------------------- | :-------------------------------------- |
| **tensorflow.js** | Yes (tf.linalg.solve) 32              | No 34                            | GPU-accelerated. User's existing framework. Robust primitives.                | **Primary Choice for Implementation**   |
| **ml-matrix**     | Yes (solve, with SVD for singular) 37 | Yes (EigenvalueDecomposition) 37 | CPU-only. Praised for robustness.8 Excellent for validation or as a fallback. | **Recommended for Validation/Fallback** |
| **numeric.js**    | Yes                                   | Yes (eig) 39                     | Documented internal errors and instability.8                                  | **Not Recommended**                     |
| **math.js**       | Yes                                   | Yes (eigs, symmetric only) 38    | eigs only for symmetric matrices. Not designed for high-precision use.38      | Not Recommended                         |

### **4.2 Implementing RQI for a Single Eigenpair**

The core of the solution is a function that implements RQI to find a single eigenpair. The algorithm is an iterative refinement process.

#### **Pseudocode for RQI**

function RayleighQuotientIteration(A, v_initial, tolerance, max_iterations):  
 v \= normalize(v_initial)  
 lambda \= 0

for k from 1 to max_iterations:  
 // 1\. Compute Rayleigh Quotient as the shift  
 lambda \= v_transpose \* A \* v

    // 2\. Form the shifted matrix
    A\_shifted \= A \- lambda \* I

    // 3\. Solve the linear system for the next vector approximation
    // This is the core computational step
    y \= solve(A\_shifted, v)

    // 4\. Normalize the new vector
    v\_new \= normalize(y)

    // 5\. Check for convergence
    if norm(v\_new \- v) \< tolerance:
      // Recompute final lambda for highest accuracy with converged v\_new
      lambda\_final \= v\_new\_transpose \* A \* v\_new
      return (lambda\_final, v\_new)

    v \= v\_new

// If max iterations reached, return the last computed pair  
 lambda_final \= v_transpose \* A \* v  
 return (lambda_final, v)

#### **TypeScript Implementation with tensorflow.js**

The following TypeScript function implements the RQI algorithm. It is essential to use tf.tidy() to manage memory, as tensorflow.js operations create new tensors that must be manually disposed of to prevent memory leaks.34

TypeScript

import \* as tf from '@tensorflow/tfjs';

interface EigenPair {  
 eigenvalue: number;  
 eigenvector: tf.Tensor1D;  
}

/\*\*  
 \* Computes a single eigenpair (eigenvalue and eigenvector) of a symmetric matrix  
 \* using the Rayleigh Quotient Iteration method.  
 \*  
 \* @param A The symmetric input matrix (tf.Tensor2D).  
 \* @param initialVector A starting vector for the iteration (tf.Tensor1D).  
 \* @param tolerance The convergence tolerance. Iteration stops when the L2 norm of the  
 \* difference between successive eigenvectors is below this value.  
 \* @param maxIterations The maximum number of iterations to perform.  
 \* @returns An object containing the computed eigenvalue and eigenvector.  
 \*/  
export function rayleighQuotientIteration(  
 A: tf.Tensor2D,  
 initialVector: tf.Tensor1D,  
 tolerance: number \= 1e-10,  
 maxIterations: number \= 100  
): EigenPair {  
 // Ensure the input matrix is square  
 if (A.shape\!== A.shape) {  
 throw new Error('Input matrix must be square.');  
 }

let v \= tf.tidy(() \=\> initialVector.div(tf.norm(initialVector))) as tf.Tensor1D;  
 let eigenvalue: number \= 0;

for (let i \= 0; i \< maxIterations; i++) {  
 const result \= tf.tidy(() \=\> {  
 // 1\. Compute Rayleigh Quotient: lambda \= v^T \* A \* v  
 const Av \= tf.matMul(A, v.as2D(v.shape, 1)).as1D();  
 const lambda \= tf.dot(v, Av).dataSync();

      // 2\. Form the shifted matrix: A\_shifted \= A \- lambda \* I
      const I \= tf.eye(A.shape);
      const A\_shifted \= A.sub(I.mul(tf.scalar(lambda)));

      // 3\. Solve the linear system: (A \- lambda\*I)y \= v
      // This is the most computationally expensive step.
      let y: tf.Tensor1D;
      try {
        y \= tf.linalg.solve(A\_shifted, v.as2D(v.shape, 1)).as1D();
      } catch (error) {
        // If the solver fails, it's likely because the matrix is singular,
        // which means we have converged to an exact eigenvalue.
        // We can break the loop and return the current result.
        console.warn('Linear solver failed, likely due to convergence. Returning last valid vector.');
        return { converged: true, v, lambda };
      }

      // 4\. Normalize the new vector: v\_new \= y / ||y||
      const v\_new \= y.div(tf.norm(y));

      // 5\. Check for convergence: ||v\_new \- v|| \< tolerance
      const diff \= tf.norm(v\_new.sub(v)).dataSync();
      const converged \= diff \< tolerance;

      return { converged, v: v\_new, lambda };
    });

    // Dispose of the old vector \`v\` before assigning the new one
    tf.dispose(v);
    v \= result.v as tf.Tensor1D;
    eigenvalue \= result.lambda;

    if (result.converged) {
      // Re-calculate the final eigenvalue with the fully converged eigenvector for max accuracy
      const finalEigenvalue \= tf.tidy(() \=\> {
        const Av \= tf.matMul(A, v.as2D(v.shape, 1)).as1D();
        return tf.dot(v, Av).dataSync();
      });
      return { eigenvalue: finalEigenvalue, eigenvector: v };
    }

}

console.warn(\`RQI did not converge within ${maxIterations} iterations.\`);  
 // Return the last computed pair if max iterations are reached  
 return { eigenvalue, eigenvector: v };  
}

### **4.3 Finding the k Smallest Eigenvectors for Spectral Clustering**

The function above finds a single eigenpair. To perform spectral clustering, we need the k eigenvectors corresponding to the k smallest eigenvalues. A robust and pragmatically simple method to achieve this is to run the RQI solver k times, each time starting with an initial vector that is orthogonal to the eigenvectors already found. This process, a form of deflation, encourages the algorithm to find new, undiscovered eigenvectors.  
The Gram-Schmidt process is used for orthogonalization. Given a new vector v and a set of already found orthonormal vectors {u1​,u2​,…,uj​}, the orthogonalized vector v′ is computed as:  
v′=v−i=1∑j​projui​​(v)=v−i=1∑j​(v⋅ui​)ui​

#### **Algorithm for k Eigenvectors**

function findKSmallestEigenpairs(A, k,...options):  
 eigenpairs \=  
 n \= dimension of A

for i from 1 to k:  
 // 1\. Generate a random starting vector  
 v_start \= random_vector(n)

    // 2\. Orthogonalize v\_start against all previously found eigenvectors
    for found\_pair in eigenpairs:
      projection \= dot(v\_start, found\_pair.eigenvector)
      v\_start \= v\_start \- projection \* found\_pair.eigenvector

    // 3\. Run RQI with the orthogonalized starting vector
    // A small shift can help target the smallest eigenvalues more reliably
    // when multiple are close to zero.
    new\_pair \= rayleighQuotientIteration(A, v\_start,...options)

    // 4\. Store the result
    eigenpairs.push(new\_pair)

// 5\. Sort the pairs by eigenvalue in ascending order  
 sort(eigenpairs by eigenvalue)

return first k pairs from sorted list

This iterative orthogonalization approach is significantly simpler to implement than full matrix deflation (A' \= A \- \\lambda v v^T) and is highly effective for matrices with well-separated eigenvalues, as is often the case for the smallest eigenvalues of the graph Laplacian. It provides a direct, step-by-step path to acquiring the necessary spectral embedding for the clustering task.

## **Section 5: Validation, Pitfalls, and Conclusion**

Implementing a new numerical algorithm requires a robust validation strategy and an awareness of potential pitfalls. This final section outlines how to verify the correctness of the RQI implementation and addresses common issues that may arise, concluding with a summary of the recommended path forward.

### **5.1 Validation Strategy**

A two-tiered validation approach is recommended to ensure both the correctness of the RQI component and its effectiveness in the overall application.

1. **Component-Level Validation:** The custom rayleighQuotientIteration function should be tested in isolation. The most reliable way to do this within the JavaScript ecosystem is to compare its output against the results from the ml-matrix library. For each of the failing test case matrices, compute the eigenpairs using both the new tensorflow.js-based RQI implementation and the EigenvalueDecomposition class from ml-matrix.37 The resulting eigenvalues and eigenvectors should match to a high degree of precision. This provides a pure JavaScript-based verification of the algorithm's correctness, independent of the full clustering pipeline.
2. **End-to-End System Validation:** The ultimate test is the successful resolution of the original problem. The new RQI-based eigensolver should be integrated into the full spectral clustering pipeline, replacing the Jacobi solver. The original suite of RBF kernel tests must then be executed. The primary success metric is achieving an Adjusted Rand Index (ARI) of ≥0.95 on all tests that were previously failing. This confirms that the improved accuracy of the eigenvectors is sufficient to produce the correct clusterings.

### **5.2 Potential Pitfalls and Mitigation**

While RQI is powerful, it is an iterative numerical method and is subject to certain practical challenges.

- **Singularity in the Linear Solve:** The core of the RQI algorithm is the linear solve step: (A−ρk​I)yk+1​=vk​. As the Rayleigh quotient ρk​ converges to a true eigenvalue λ, the matrix on the left-hand side, A−ρk​I, becomes singular or very close to singular (ill-conditioned).40 A standard linear solver may fail or produce  
  NaN values when faced with a singular matrix.41
  - **Mitigation 1: Graceful Failure:** As shown in the TypeScript implementation, the tf.linalg.solve call can be wrapped in a try...catch block. A failure in the solver is a strong indicator that ρk​ is an excellent approximation of an eigenvalue. In this case, the algorithm can be designed to terminate gracefully, returning the last successfully computed eigenvector. This is often sufficient as the eigenvector will have already converged to high precision.
  - **Mitigation 2: SVD-based Solvers (Conceptual):** Some advanced linear algebra packages, like ml-matrix when used with the useSVD=true flag, can handle singular systems by finding a least-squares solution.37 While  
    tensorflow.js's solve does not offer this specific option, it is a valuable concept. It demonstrates that a solution can still be found, and a failure in a standard solver is a sign of convergence, not a fatal error.
  - **Mitigation 3: Regularization/Perturbation:** If singularity causes persistent issues, a tiny amount of regularization can be added to the diagonal of the matrix, for example, by shifting with ρk​−ϵ instead of ρk​, where ϵ is a very small number. This nudges the matrix away from perfect singularity, often allowing the solver to proceed. This should be used as a last resort, as it can slightly affect the final accuracy.
- **Convergence to an Unwanted Eigenvector:** RQI converges to the eigenpair "closest" to the initial guess. While it has excellent local convergence, its global behavior can be unpredictable; a random starting vector may lead to convergence to an eigenpair other than the one desired.42
  - **Mitigation:** The strategy outlined in Section 4.3—iteratively finding eigenpairs and orthogonalizing the next starting vector against all previously found eigenvectors—is the primary defense against this. By removing components of known eigenvectors from the starting vector, the iteration is forced to explore new directions in the vector space, significantly increasing the likelihood of finding a new, distinct eigenvector. Running the k-eigenvector search multiple times with different random seeds can provide further confidence that the true smallest eigenpairs have been found.

### **5.3 Conclusion and Path Forward**

The analysis has shown that the accuracy limitations of the Jacobi eigensolver are the definitive root cause of the failures in the spectral clustering RBF tests. The small numerical deviations it produces are sufficient to corrupt the spectral embedding and lead to incorrect cluster assignments.  
The recommended solution is to replace the Jacobi method with a custom implementation of **Rayleigh Quotient Iteration (RQI)**. This algorithm represents the most pragmatic and powerful choice, offering an ideal balance of extreme accuracy, rapid convergence, and manageable implementation complexity. Its cubic convergence rate for symmetric matrices will provide the high-precision eigenvectors needed to satisfy the ARI ≥0.95 requirement, and its core dependency on a linear system solve aligns perfectly with the capabilities of the existing tensorflow.js framework.  
The following checklist provides a clear, actionable path to implementing this solution:

1. **Implement the Core RQI Function:** Create the rayleighQuotientIteration function in TypeScript using tensorflow.js as detailed in Section 4.2. Pay close attention to memory management with tf.tidy() and include error handling for the linear solver.
2. **Develop the Multi-Eigenpair Wrapper:** Implement the higher-level function that finds the k smallest eigenpairs by iteratively calling the core RQI function. Use the Gram-Schmidt process to orthogonalize each new starting vector against the set of previously discovered eigenvectors.
3. **Integrate and Test:** Replace the existing Jacobi solver call within the spectral clustering algorithm with a call to the new RQI-based solver.
4. **Validate the Solution:** Perform component-level validation of the RQI function against ml-matrix and conduct end-to-end system validation by running the full test suite to confirm that all RBF tests now pass with the required ARI score.

By following this blueprint, the current numerical bottleneck can be effectively resolved, resulting in a more robust, accurate, and reliable spectral clustering implementation.

#### **Works cited**

1. Robust and Efficient Computation of Eigenvectors in a Generalized Spectral Method for Constrained Clustering \- Proceedings of Machine Learning Research, accessed on July 21, 2025, [http://proceedings.mlr.press/v54/jiang17b/jiang17b.pdf](http://proceedings.mlr.press/v54/jiang17b/jiang17b.pdf)
2. How to compute smallest non-zero eigenvalue \- Stack Overflow, accessed on July 21, 2025, [https://stackoverflow.com/questions/36060975/how-to-compute-smallest-non-zero-eigenvalue](https://stackoverflow.com/questions/36060975/how-to-compute-smallest-non-zero-eigenvalue)
3. 2.9. Computing Eigenvalues and Eigenvectors: the Power Method, and a bit beyond — Numerical Methods and Analysis with Python \- Brenton LeMesurier, College of Charleston, accessed on July 21, 2025, [https://lemesurierb.people.charleston.edu/numerical-methods-and-analysis-python/main/eigenproblems-python.html](https://lemesurierb.people.charleston.edu/numerical-methods-and-analysis-python/main/eigenproblems-python.html)
4. Solving eigenproblems with Neural Networks \- mediaTUM, accessed on July 21, 2025, [https://mediatum.ub.tum.de/doc/1632870/kwm0n4od0og42tg17pewgnx5s.pdf](https://mediatum.ub.tum.de/doc/1632870/kwm0n4od0og42tg17pewgnx5s.pdf)
5. a comparison of numerical implementations of the eigenstate expansion method for quantum molecular dynamic, accessed on July 21, 2025, [http://webdoc.sub.gwdg.de/ebook/ah/2000/ethz/tech-reports/3xx/333.pdf](http://webdoc.sub.gwdg.de/ebook/ah/2000/ethz/tech-reports/3xx/333.pdf)
6. Time comparison between Jacobi-Davidson, Lanczos and FEAST method on 4, 8, 16 and 32 nodes of the HPC cluster for the calculation of 8 energy eigenstates \- ResearchGate, accessed on July 21, 2025, [https://www.researchgate.net/figure/Time-comparison-between-Jacobi-Davidson-Lanczos-and-FEAST-method-on-4-8-16-and-32_fig4_274699843](https://www.researchgate.net/figure/Time-comparison-between-Jacobi-Davidson-Lanczos-and-FEAST-method-on-4-8-16-and-32_fig4_274699843)
7. Comparison of Numerical Methods and Open-Source Libraries for Eigenvalue Analysis of Large-Scale Power Systems \- MDPI, accessed on July 21, 2025, [https://www.mdpi.com/2076-3417/10/21/7592](https://www.mdpi.com/2076-3417/10/21/7592)
8. Exception calculating eigenvalues in numeric.js \- Stack Overflow, accessed on July 21, 2025, [https://stackoverflow.com/questions/49336606/exception-calculating-eigenvalues-in-numeric-js](https://stackoverflow.com/questions/49336606/exception-calculating-eigenvalues-in-numeric-js)
9. Chapter 7 Lanczos Methods, accessed on July 21, 2025, [https://math.ntnu.edu.tw/\~min/matrix_comp/chap7_Lanczos.pdf](https://math.ntnu.edu.tw/~min/matrix_comp/chap7_Lanczos.pdf)
10. Power iteration \- Wikipedia, accessed on July 21, 2025, [https://en.wikipedia.org/wiki/Power_iteration](https://en.wikipedia.org/wiki/Power_iteration)
11. Lecture 27\. Rayleigh Quotient, Inverse Iteration, accessed on July 21, 2025, [https://www.cs.cmu.edu/afs/cs/academic/class/15859n-f16/Handouts/TrefethenBau/RayleighQuotient-27.pdf](https://www.cs.cmu.edu/afs/cs/academic/class/15859n-f16/Handouts/TrefethenBau/RayleighQuotient-27.pdf)
12. Inverse iteration \- Wikipedia, accessed on July 21, 2025, [https://en.wikipedia.org/wiki/Inverse_iteration](https://en.wikipedia.org/wiki/Inverse_iteration)
13. How to compute the smallest eigenvalue using the power iteration algorithm?, accessed on July 21, 2025, [https://math.stackexchange.com/questions/271864/how-to-compute-the-smallest-eigenvalue-using-the-power-iteration-algorithm](https://math.stackexchange.com/questions/271864/how-to-compute-the-smallest-eigenvalue-using-the-power-iteration-algorithm)
14. Optimisation-Python/EigenValuesAndWhereToFindThem.ipynb at master \- GitHub, accessed on July 21, 2025, [https://github.com/gnthibault/Optimisation-Python/blob/master/EigenValuesAndWhereToFindThem.ipynb](https://github.com/gnthibault/Optimisation-Python/blob/master/EigenValuesAndWhereToFindThem.ipynb)
15. Shifted inverse iteration \- Wikiversity, accessed on July 21, 2025, [https://en.wikiversity.org/wiki/Shifted_inverse_iteration](https://en.wikiversity.org/wiki/Shifted_inverse_iteration)
16. 8.3. Inverse iteration — Fundamentals of Numerical Computation \- Toby Driscoll, accessed on July 21, 2025, [https://tobydriscoll.net/fnc-julia/krylov/inviter.html](https://tobydriscoll.net/fnc-julia/krylov/inviter.html)
17. The Inverse Power Method \- arnold@uark.edu, accessed on July 21, 2025, [https://arnold.hosted.uark.edu/NLA/Pages/invpower.pdf](https://arnold.hosted.uark.edu/NLA/Pages/invpower.pdf)
18. Inverse Iteration \- Netlib.org, accessed on July 21, 2025, [https://www.netlib.org/utk/people/JackDongarra/etemplates/node96.html](https://www.netlib.org/utk/people/JackDongarra/etemplates/node96.html)
19. ALAFF The Rayleigh Quotient Iteration \- UT Computer Science, accessed on July 21, 2025, [https://www.cs.utexas.edu/\~flame/laff/alaff/chapter09-Rayleigh-quotient-iteration.html](https://www.cs.utexas.edu/~flame/laff/alaff/chapter09-Rayleigh-quotient-iteration.html)
20. Rayleigh quotient iteration \- Wikipedia, accessed on July 21, 2025, [https://en.wikipedia.org/wiki/Rayleigh_quotient_iteration](https://en.wikipedia.org/wiki/Rayleigh_quotient_iteration)
21. Rayleigh Quotient Iteration Explained \- Number Analytics, accessed on July 21, 2025, [https://www.numberanalytics.com/blog/rayleigh-quotient-iteration-explained](https://www.numberanalytics.com/blog/rayleigh-quotient-iteration-explained)
22. Lanczos algorithm \- Wikipedia, accessed on July 21, 2025, [https://en.wikipedia.org/wiki/Lanczos_algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm)
23. Lanczos Algorithm: A Comprehensive Guide \- Number Analytics, accessed on July 21, 2025, [https://www.numberanalytics.com/blog/lanczos-algorithm-linear-algebra-engineering-mathematics](https://www.numberanalytics.com/blog/lanczos-algorithm-linear-algebra-engineering-mathematics)
24. Mastering Lanczos Algorithm in Linear Algebra \- Number Analytics, accessed on July 21, 2025, [https://www.numberanalytics.com/blog/lanczos-algorithm-linear-algebra-guide](https://www.numberanalytics.com/blog/lanczos-algorithm-linear-algebra-guide)
25. Lanczos Method A. Ruhe \- NetLib.org, accessed on July 21, 2025, [https://www.netlib.org/utk/people/JackDongarra/etemplates/node103.html](https://www.netlib.org/utk/people/JackDongarra/etemplates/node103.html)
26. Quality of eigenvalue approximation in Lanczos method, accessed on July 21, 2025, [https://scicomp.stackexchange.com/questions/23536/quality-of-eigenvalue-approximation-in-lanczos-method](https://scicomp.stackexchange.com/questions/23536/quality-of-eigenvalue-approximation-in-lanczos-method)
27. Thick-Restart Lanczos Method for Symmetric Eigenvalue Problems \- eScholarship.org, accessed on July 21, 2025, [https://escholarship.org/content/qt87z3114q/qt87z3114q.pdf](https://escholarship.org/content/qt87z3114q/qt87z3114q.pdf)
28. Mastering Lanczos Algorithm for Computational Chemistry \- Number Analytics, accessed on July 21, 2025, [https://www.numberanalytics.com/blog/mastering-lanczos-algorithm-computational-chemistry](https://www.numberanalytics.com/blog/mastering-lanczos-algorithm-computational-chemistry)
29. An application of the Lanczos Method \- Drexel University, accessed on July 21, 2025, [http://www.physics.drexel.edu/\~bob/Term_Reports/Hoppe_02.pdf](http://www.physics.drexel.edu/~bob/Term_Reports/Hoppe_02.pdf)
30. The use of Lanczos's method to solve the large generalized symmetric definite eigenvalue problem \- ResearchGate, accessed on July 21, 2025, [https://www.researchgate.net/publication/24298087_The_use_of_Lanczos's_method_to_solve_the_large_generalized_symmetric_definite_eigenvalue_problem](https://www.researchgate.net/publication/24298087_The_use_of_Lanczos's_method_to_solve_the_large_generalized_symmetric_definite_eigenvalue_problem)
31. Lanczos Algorithm with SI. \- Netlib.org, accessed on July 21, 2025, [https://www.netlib.org/utk/people/JackDongarra/etemplates/node171.html](https://www.netlib.org/utk/people/JackDongarra/etemplates/node171.html)
32. tf.linalg.solve | TensorFlow v2.16.1, accessed on July 21, 2025, [https://www.tensorflow.org/api_docs/python/tf/linalg/solve](https://www.tensorflow.org/api_docs/python/tf/linalg/solve)
33. Inverse Power Method Explained \- Number Analytics, accessed on July 21, 2025, [https://www.numberanalytics.com/blog/inverse-power-method-explained-matrix-computations](https://www.numberanalytics.com/blog/inverse-power-method-explained-matrix-computations)
34. Linear Regression with TensorFlow.js \- Eric Jinks, accessed on July 21, 2025, [https://ericjinks.com/blog/2018/linear-regression-with-tensorflow-js/](https://ericjinks.com/blog/2018/linear-regression-with-tensorflow-js/)
35. glMatrix, accessed on July 21, 2025, [https://glmatrix.net/](https://glmatrix.net/)
36. Introduction to Tensorflow.js \- DEV Community, accessed on July 21, 2025, [https://dev.to/eteimz/introduction-to-tensorflowjs-443h](https://dev.to/eteimz/introduction-to-tensorflowjs-443h)
37. ml-matrix \- npm, accessed on July 21, 2025, [https://www.npmjs.com/package/ml-matrix](https://www.npmjs.com/package/ml-matrix)
38. Function eigs \- Math.js, accessed on July 21, 2025, [https://mathjs.org/docs/reference/functions/eigs.html](https://mathjs.org/docs/reference/functions/eigs.html)
39. Numeric Javascript: Documentation, accessed on July 21, 2025, [https://ccc-js.github.io/numeric2/documentation.html](https://ccc-js.github.io/numeric2/documentation.html)
40. A stable simultaneous vector inverse iteration method with shift \- ResearchGate, accessed on July 21, 2025, [https://www.researchgate.net/publication/237895903_A_stable_simultaneous_vector_inverse_iteration_method_with_shift](https://www.researchgate.net/publication/237895903_A_stable_simultaneous_vector_inverse_iteration_method_with_shift)
41. Finding k-smallest eigen values and its corresponding eigen vector for large matrix, accessed on July 21, 2025, [https://stackoverflow.com/questions/55379347/finding-k-smallest-eigen-values-and-its-corresponding-eigen-vector-for-large-mat](https://stackoverflow.com/questions/55379347/finding-k-smallest-eigen-values-and-its-corresponding-eigen-vector-for-large-mat)
42. A Complex-Projected Rayleigh Quotient Iteration for Targeting Interior Eigenvalues | SIAM Journal on Matrix Analysis and Applications, accessed on July 21, 2025, [https://epubs.siam.org/doi/10.1137/23M1622155](https://epubs.siam.org/doi/10.1137/23M1622155)

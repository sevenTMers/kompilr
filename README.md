#  Domain Specific Compilers for Drug Discovery :electron:

> The next decade will see a Cambrian explosion of novel computer architectures... [A New Golden Age For Computer Architecture](https://cacm.acm.org/magazines/2019/2/234352-a-new-golden-age-for-computer-architecture/fulltext)

![The First Compiler](images/holy_mountain.gif)

```
So let it not look strange 
if I claim
that it is much easier to explain 
the movement of the giant celestial bodies 
than to interpret in mechanical terms 
the origination of just a single caterpillar
or a tiny grass.
``` 

> Kant, *Natural History and the Theory of Heaven*, 1755

---

Machine Learning for Drug Discovery has [made many strides](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7577280/) over the last decade. Yet there is a long, long way to go. 
 
# Molecular Dynamics 

It is said Molecular Dynamics begin with 1959's [Studies in Molecular Dynamics](https://github.com/sevenTMers/kompilr/blob/main/papers/1959_Studies_in_Molecular%20Dynamics.pdf). 

# Protein Folding

 > Structure-function relationships are the fundamental object of knowledge in protein chemistry; they allow us to rationally design drugs, engineer proteins with new functions, and understand why mutations cause disease. [- On The Origin of Proteins](https://www.chemistryworld.com/features/on-the-origin-of-proteins/3004719.article)

> There is now a testable explanation for how a protein can fold so quickly: A protein solves its large global optimization problem as a series of smaller local optimization problems, growing and assembling the native structure from peptide fragments, local structures first. [- The Protein Folding Problem](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2443096/)

![Simpsons](images/simpsons.gif)

We used to believe that the protein folding problem is comprised of three closely related puzzles:
* (a) What is the folding code? 
* (b) What is the folding mechanism?
* (c) Can we predict the native structure of a protein from its amino acid sequence? [source](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2443096/)


## Data Sources


[CATH/Gene3D](https://www.cathdb.info/) - 151 Million Protein Domains Classified into 5,481 Superfamilies

[NCBI Conserved Domains Database](https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi) - resource for the annotation of functional units in proteins

[Protein Data Bank](https://www.rcsb.org/)

[Scop 2](https://scop.mrc-lmb.cam.ac.uk/) - Structural Classification of Proteins

[UniProt](https://www.uniprot.org/) -  comprehensive, high-quality and freely accessible resource of protein sequence and functional information.

[Fold@Home](https://foldingathome.org/about/)

## Deep Learning Protein Folding

### [AlphaFold 14](https://www.predictioncenter.org/casp14/doc/presentations/2020_12_01_TS_predictor_AlphaFold2.pdf)

  * [:computer: Code](https://github.com/deepmind/alphafold)
  * [:book: Paper 2](https://www.nature.com/articles/s41586-021-03819-2_reference.pdf)
  * [:book: Paper](https://www.nature.com/articles/s41586-019-1923-7.epdf?author_access_token=Z_KaZKDqtKzbE7Wd5HtwI9RgN0jAjWel9jnR3ZoTv0MCcgAwHMgRx9mvLjNQdB2TlQQaa7l420UCtGo8vYQ39gg8lFWR9mAZtvsN_1PrccXfIbc6e-tGSgazNL_XdtQzn1PHfy21qdcxV7Pw-k3htw%3D%3D)
  * [:newspaper: article](https://deepmind.com/blog/article/AlphaFold-Using-AI-for-scientific-discovery)
  * [AlpahFold 14 Results Discussion](https://dasher.wustl.edu/bio5357/discussion/oxford-alphafold2.pdf)
  * [What AlphaFold means for Structural BioInformatics](https://ammiellewb.medium.com/what-alphafold-means-for-structural-bioinformatics-78117adb7d11)
  * [AlphaFold 2 Explained](https://youtu.be/B9PL__gVxLI) - Yanick Video
  * [Illustrated Transformer](kjalammar.github.io/illustrated-transformer/)
  * [Transformers from Scratch](http://peterbloem.nl/blog/transformers)

### [AlphaFold 13](https://www.predictioncenter.org/CASP13/doc/presentations/Pred_CASP13-Structure-AlphaFold-Jumper.pdf)

  * [:floppy_disk: Code](https://github.com/deepmind/deepmind-research/tree/master/alphafold_casp13)
  * [:floppy_disk: Code](https://github.com/dellacortelab/prospr) - Prospr - Open Source Implementation
  * [:book: Prospr Paper](https://www.biorxiv.org/content/10.1101/830273v1) 
  * [AlphaFold @ Casp13: What Just Happened?](https://moalquraishi.wordpress.com/2018/12/09/alphafold-casp13-what-just-happened/) 

### [MiniFold](https://github.com/hypnopump/MiniFold) - Open Source toy example of AlphaFold 13 algorithm 

> The DeepMind work presented @ CASP was not a technological breakthrough (they did not invent any new type of AI) but an engineering one: they applied well-known AI algorithms to a problem along with lots of data and computing power and found a great solution through model design, feature engineering, model ensembling and so on...

> Based on the premise exposed before, the aim of this project is to build a model suitable for protein 3D structure prediction inspired by AlphaFold and many other AI solutions that may appear and achieve SOTA results.

![MiniFold](minifold.png)

> Two different residual neural networks (ResNets) are used to predict angles between adjacent aminoacids (AAs) and distance between every pair of AAs of a protein. For distance prediction a 2D Resnet was used while for angles prediction a 1D Resnet was used.

### PDNet

> As deep learning algorithms drive the progress in protein structure prediction, a lot remains to be studied at this merging superhighway of deep learning and protein structure prediction. Recent findings show that inter-residue distance prediction, a more granular version of the well-known contact prediction problem, is a key to predicting accurate models. However, deep learning methods that predict these distances are still in the early stages of their development. To advance these methods and develop other novel methods, a need exists for a small and representative dataset packaged for faster development and testing. In this work, we introduce protein distance net (PDNET), a framework that consists of one such representative dataset along with the scripts for training and testing deep learning methods. The framework also includes all the scripts that were used to curate the dataset, and generate the input features and distance maps.

[:desktop: Github](https://github.com/ba-lab/pdnet/)

[:book: Paper](https://www.nature.com/articles/s41598-020-70181-0) 

[:vhs: YouTube](https://youtu.be/uAIuA1O7iE8)

## TPU

[A Domain Specific Supercomputer for Training Deep Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3360307)

* 128x128- or 256x256-element systolic arrays of multipliers per core
* [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
* TPUv2 uses a 16x16 2D torus network topology.
* XLA compiler optimizations 

> Benchmarks suggests the TPUv3 chip performs similarly to the contemporary Volta GPU chip, but parallel scaling for production applications is stronger for the TPUv3 supercomputer

## [Bfloat16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

Designed to be used in hardware accelerating machine learning algorithms. 

Same formate as IEEE 754 single precision floating point BUT truncates
mantissa from 23 -> 7 bits. (floating point is made up of sign, exponent, and
mantissa) 

> Neural networks are far more sensitive to the size of the exponent than that of the mantissa
[source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

> Deep Learning models are known to tolerate lower numerical precision...the network can accomplish a task with the same accuracy using a lower precision approximation.
[source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus1)

> Surprisingly some models can even reach a higher accuracy with lower precision [source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

Has exactly same range as float32 - ~1e-38 to ~3e38

* 1 sign bit
* 8 exponent bits
* 7 mantissa bits 

SEEEEEEEEMMMMMMM

> Of deep learning models...the bfloat16 format works as well as the FP32 format while delivering increased performance and reducing memory usage. [source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) 

> The physical size of a hardware multiplier scales with the square of the mantissa width. With fewer mantissa bits than FP16, the bfloat16 multipliers are about half the size in silicon of a typical FP16 multiplier, and they are eight times smaller than an FP32 multiplier!
[source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) 

> We typically recommend keeping weights and gradients in FP32 but converting activations to bfloat 16 [source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

## XLA

The CPU and GPU backends included with XLA use [LLVM](http://llvm.org) for low-level IR, optimization, and code-generation.

[XLA - HLO](https://www.tensorflow.org/xla/architecture#how_does_xla_work)

## MLIR

[MLIR - HLO](https://github.com/tensorflow/mlir-hlo)

MLIR is:

> intended for easy expression and optimization of computations involving deep loop nests and dense matrices of high dimensionality.

It was presented in [MLIR: A Compiler Infrastructure for the End ofMoore’s Law](https://arxiv.org/pdf/2002.11054.pdf)

MLIR is a Static Single Assignment form (SSA) compiler:

* SSA is a property of an intermediate representation (IR), which requires that
  * each variable be assigned exactly once
  * every variable be defined before it is used
* In an SSA, existing variables in the original IR are split into versions, new variables typically indicated by the original name with a subscript, so that every definition gets its own version.

MLIR introduces concepts from the [polytype model](https://en.wikipedia.org/wiki/Polytope_model) for loop optimization.

(In compiler construction, a [basic block](https://en.wikipedia.org/wiki/Basic_block) is a straight-line code sequence with no branches in except to the entry and no branches out except at the exit)

MLIR has no fixed/built-in list of globally known operations (no “intrinsics”)


## LLVM

> the LLVM libraries have many capabilities, but they don't actually do anything by themselves. It is up to the designer of the client of the libraries (e.g., the Clang C compiler) to decide how to put the pieces to best use. 

> This careful layering, factoring, and focus on subset-ability is also why the LLVM optimizer can be used for such a broad range of different applications in different contexts. 

[Architecture of Open Source Applications LLVM Chapter](http://www.aosabook.org/en/llvm.html)

* Global identifiers (functions, global variables) begin with the '@' character
* Local identifiers (register names, types) begin with the '%' character


### Other Links

Intel [Foundry Services FactSheet](https://newsroom.intel.com/wp-content/uploads/sites/11/2021/03/intel-foundry-services-fact-sheet-229940.pdf)

Amazon [EC2 F1 Instances](https://aws.amazon.com/ec2/instance-types/f1/) - FPGA accelerator development and deployment in the cloud



## Machine Learning the Schrodinger Equation

<img src="qsim.png" width="800" height="500" /> 

[source](https://youtu.be/_bdvpmleAgw)

## [Solving the Quantum Many-Body Problem with Artificial Neural Networks (2016)](https://arxiv.org/pdf/1606.02318.pdf)

* 💻 [code version from the authors](https://gitlab.com/nqs)

* 💻 [version of the paper in code](https://github.com/zeldredge/py-nqs)

* 💻 [version of the paper in Jupyter Notebook](https://github.com/bartolsthoorn/NQS-numpy/blob/master/NQS.ipynb)

> In principle, an exponential amount of information is needed to fully encode a generic [many-body quantum state](https://youtu.be/uTCeQHzQMdc).

> However, Nature often proves herself benevolent, and a [wave function](https://phys.libretexts.org/Bookshelves/University_Physics/Book%3A_University_Physics_(OpenStax)/Book%3A_University_Physics_III_-_Optics_and_Modern_Physics_(OpenStax)/07%3A_Quantum_Mechanics/7.02%3A_Wavefunctions) representing a physical many-body system can
be typically characterized by an amount of information much smaller than the maximum capacity of the corresponding [Hilbert space](https://youtu.be/g-eNeXlZKAQ). A limited amount of [quantum entanglement](https://www.quantamagazine.org/entanglement-made-simple-20160428/), as well as the typicality of a small number of physical states, are then the blocks on which modern approaches build upon to solve the many-body [Schrödinger’s equation](https://plus.maths.org/content/schrodinger-1) with a limited amount of classical resources.

<img src="quantum_ann.png" width="60%" height="50%" />


# Resources 

[A Tourists Guide to LLVM](https://blog.regehr.org/archives/1453)

[The Architecture of Open Source Applications: LLVM](http://www.aosabook.org/en/llvm.html) - Chris Laetner

[LLVM for Graduate Students](http://www.cs.cornell.edu/~asampson/blog/llvm.html)

[Memory Models Underlying Programming Languages](http://canonical.org/~kragen/memory-models/)

[Compiler Gym](https://github.com/facebookresearch/CompilerGym)

[Apache TVM](https://tvm.apache.org/) - Machine Learning Compiler Framework

[Tiramisu Compiler](http://tiramisu-compiler.org/)

[Dive Into Deep Learning Compiler](https://tvm.d2l.ai/index.html)

[Nvidia NVCC Compiler](https://developer.nvidia.com/cuda-llvm-compiler)

[Halide](https://halide-lang.org/)

# Companies

[Acellera](https://www.acellera.com/) - Hardware/Software Molecular Dynamics

[Cerebras](https://cerebras.net/industries/health-and-pharma/)

[GraphCore](https://www.graphcore.ai/healthcare)

[GSI Technology](https://www.gsitechnology.com/Hardware-Accelerated-Search-for-Drug-Discovery) - Molecular Similarity

[Pharmacelera](https://new.pharmacelera.com/)

# Research Groups

[**ATOM**](https://atomscience.org/) - *'Transforming Drug Discovery'* - HPC

[COBRE Center for Targeted Therapeutics](https://sc.edu/study/colleges_schools/pharmacy/centers/cobre_center_for_targeted_therapeutics/index.php) - University of South Carolina


# References

(2021) [High Througput Virtual Laboratory For Drug Discovery Using Massive Datasets](https://github.com/kompilr/kompilR/blob/main/papers/2021_high-throughput_virtual_laboratory_for_drug_discovery_using_massive_datasets.pdf)

(2021) [AI Powered Compiler Techniques for DL Code Optimization](https://arxiv.org/pdf/2104.05573.pdf)

(2021) [A MLIR Dialect for Quantum Assembly Languages](https://arxiv.org/pdf/2101.11365.pdf)

(2021) [Ten Lessons From Three Generations Shaped Google’s TPUv4i](https://conferences.computer.org/iscapub/pdfs/ISCA2021-4ghucdBnCWYB7ES2Pe4YdT/333300a001/333300a001.pdf)

(2021) [Cortex: A Compiler for Recursive Deep Learning Models](https://arxiv.org/pdf/2011.01383.pdf)

(2020) [MDBenchmark:A toolkit to optimize the performance of molecular dynamics simulations](https://aip.scitation.org/doi/10.1063/5.0019045)

(2020) [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/abs/2002.03794)

(2020) ["Ab-Initio Solution of the Many-Electron Schroedinger Equation with Deep Neural Networks"](https://arxiv.org/pdf/1909.02487.pdf) - FermiNet

(2020) [Deep neural network solution of the electronic Schrödinger equation](https://arxiv.org/pdf/1909.08423.pdf) - PauliNet

(2020) [Variational Principles in Quantum Monte Carlo: The Troubled Story of Variance Minimization](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00147)

(2020) [Fermionic neural-network states for ab-initio electronic structure](https://www.nature.com/articles/s41467-020-15724-9)

(2020) [Data Driven Science & Engineering:Machine Learning, Dynamical Systems and Control](http://databookuw.com/)

(2020) [MLIR: A Compiler Infrastructure for the End ofMoore’s Law](https://arxiv.org/pdf/2002.11054.pdf)

(2019) [Quantum Entanglement in Deep Learning Architectures](https://arxiv.org/pdf/1803.09780.pdf)

(2019) [Fully Integrated On-FPGA Molecular Dynamics Simulations](https://arxiv.org/abs/1905.05359)

(2019) [Relay: A High-Level Compiler for Deep Learning](https://arxiv.org/pdf/1904.08368.pdf)

(2018) [Machine Learning in Compiler Optimization](https://zwang4.github.io/publications/pieee18.pdf)

(2018) [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/pdf/1802.01528.pdf)

(2018) [The Simple Essence of Automatic Differentiation](http://conal.net/papers/essence-of-ad/essence-of-ad-icfp.pdf)

(2017) [In-Datacenter Performance Analysis of a Tensor Processing Unit​](https://drive.google.com/file/d/0Bx4hafXDDq2EMzRNcy1vSUxtcEk/view?resourcekey=0-ulCsvFTNky29UIPJ3pHyCw)

(2016) [Solving the Quantum Many-Body Problem with Artificial Neural Networks](https://arxiv.org/pdf/1606.02318.pdf)

(2015) [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671)

(2015) [Automatic Differentiation in Machine Learning: A Survey](https://arxiv.org/pdf/1502.05767.pdf)

(2014) [Anton 2: Raising the bar for performance and programmability in a special-purpose molecular dynamics supercomputer](https://github.com/sevenTMers/kompilr/blob/main/papers/2014_Anton2_Raising_Bar_Special_Purpose_Molecular_Dynamics_SuperComputer.pdf)

(2012) [A Systolic Array-Based FPGA Parallel Architecture for the BLAST Algorithm](https://www.hindawi.com/journals/isrn/2012/195658/)

(2011) [A high performance implementation for Molecular Dynamics simulations on a FPGA supercomputer](https://github.com/sevenTMers/kompilr/blob/main/papers/2011_high_performance_implementation_molecular_dynamics_fpga.pdf)

(2008) [Anton, A Special-Purpose Machine For Molecular Dynamics Simulation](https://cacm.acm.org/magazines/2008/7/5372-anton-a-special-purpose-machine-for-molecular-dynamics-simulation/fulltext)

(2008) [High-Throughput Pairwise Point Interactions in Anton, a Specialized Machine for Molecular Dynamics Simulation](https://github.com/sevenTMers/kompilr/blob/main/papers/2008_High_Throughput_Pairwise_Point_Interactions_in_Anton.pdf)

(1982) [Why Systolic Architectures?](https://course.ece.cmu.edu/\~ece740/f13/lib/exe/fetch.php?media=kung_-_1982_-_why_systolic_architectures.pdf)

(1981) [Trace Scheduling: A Technique for Global Microcode Compaction](https://people.eecs.berkeley.edu/\~kubitron/courses/cs252-S12/handouts/papers/TraceScheduling.pdf)

(1959) [Studies in Molecular Dynamics](https://github.com/sevenTMers/kompilr/blob/main/papers/1959_Studies_in_Molecular%20Dynamics.pdf)


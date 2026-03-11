# CHANGELOG


## v1.0.0 (2026-02-17)

### Bug Fixes

- **generate**: Remove ambiguities in problem generation ([`5339d04`](https://github.com/atomwalk12/linalg-zero/commit/5339d04bd7f1a8253e744a209f47179f67a47394))
- **distillation**: Remove vllm chat templates ([`57be920`](https://github.com/atomwalk12/linalg-zero/commit/57be920fdb2ca9524593b67b735dfee8716affc0))
- **grpo**: Adjust launch script parameters ([`a40be0e`](https://github.com/atomwalk12/linalg-zero/commit/a40be0ebcf73657a4d0c695ba933d379a31eef2b))
- **grpo**: Log statistics ([`b983abd`](https://github.com/atomwalk12/linalg-zero/commit/b983abd1760b26b62c2e101ac76e1e3290359ec9))
- **sft**: Adjust optional accuracy callback ([`16b7d32`](https://github.com/atomwalk12/linalg-zero/commit/16b7d322f5a50fd830c1ea5acd51e3ca20702454))

### Documentation

- **config**: Add sample config files ([`62290aa`](https://github.com/atomwalk12/linalg-zero/commit/62290aa84d011ef52078272d241204518a78cf16))
- **report**: Finalize project ([`ffbfcb2`](https://github.com/atomwalk12/linalg-zero/commit/ffbfcb24a0cc54bcd888b4b7c19148e75ffce4f7))

### Features

- **distillation**: Add data generation configs ([`6aab283`](https://github.com/atomwalk12/linalg-zero/commit/6aab283a343c6273bfdc274e7397dff76bb3fa5c))
- **distillation**: Add validation script ([`8344dba`](https://github.com/atomwalk12/linalg-zero/commit/8344dbad775768a7a74ffd83dec56587514e1c49))
- **distillation**: Improve multi-turn generation with progress tracking ([`f58eadf`](https://github.com/atomwalk12/linalg-zero/commit/f58eadf0083f135f2fee2405b43eca1c2705d84e))
- **grpo**: Add environment abstraction layer with base classes ([`edc9615`](https://github.com/atomwalk12/linalg-zero/commit/edc9615163385df75ad18aa15f918e534c45f7fa))
- **grpo**: Add linear algebra environment with reward computation ([`4e46e7c`](https://github.com/atomwalk12/linalg-zero/commit/4e46e7c9e0040ee34e5fca060f652272f0c6ed1f))
- **grpo**: Add linear algebra tools for matrix operations ([`d2f08b1`](https://github.com/atomwalk12/linalg-zero/commit/d2f08b1231c2c0f493bbea1f3f6718cb6829c0fa))
- **grpo**: Add reward model and RL training utilities ([`aba4bae`](https://github.com/atomwalk12/linalg-zero/commit/aba4baefdbbe9c21c01ed82d2a062f788062f9d6))
- **grpo**: Add task selection and training infrastructure ([`d82f8c6`](https://github.com/atomwalk12/linalg-zero/commit/d82f8c69506056c227825741ae7098eabbcb2067))
- **grpo**: Add tool calling agent implementation ([`b95ec50`](https://github.com/atomwalk12/linalg-zero/commit/b95ec50ce7956f897a1bde16481633cb2395c77b))
- **grpo**: Add training and evaluation scripts ([`38b99fd`](https://github.com/atomwalk12/linalg-zero/commit/38b99fd20ecf08d0fa7ce185b6094b1d782d90d9))
- **grpo**: Add utility modules for training infrastructure ([`7362699`](https://github.com/atomwalk12/linalg-zero/commit/73626995bdcfd365b40fcd6813eca5ee99f2b8c0))
- **grpo**: Add yaml config files ([`bcb5b68`](https://github.com/atomwalk12/linalg-zero/commit/bcb5b68ef026110a8fcee14da23e5bf4736c8251))
- **sft**: Add model evaluation and dataset preparation scripts ([`5e8ee4c`](https://github.com/atomwalk12/linalg-zero/commit/5e8ee4c9d8fdf90e825a8a907216dadf78e8103b))
- **sft**: Add yaml training configs ([`65ff234`](https://github.com/atomwalk12/linalg-zero/commit/65ff2348dc592d790bef130ecf593ce157ebdaf3))
- **sft**: Adjust tool evaluation callback ([`c7f2c9d`](https://github.com/atomwalk12/linalg-zero/commit/c7f2c9daf2b8506e3687bd92d1d90d6599c99aa7))
- **sft**: Improve tool calling accuracy callback with weave logging ([`823b263`](https://github.com/atomwalk12/linalg-zero/commit/823b263a447470a30993c809264dbb20494d1331))
- **sft**: Refactor diagnostics and callbacks and improve evaluation logging ([`10ae836`](https://github.com/atomwalk12/linalg-zero/commit/10ae836eb0c7af9b7756bc0690c4461f2eaf3c8d))
- **system_prompts**: Add SFT system prompt and improve tool usage guidelines ([`061d21a`](https://github.com/atomwalk12/linalg-zero/commit/061d21aa6de9dc5e0f5ff3ac97a8f71be7a83c2e))

### Refactoring

- Reorganize training entry points ([`5a65164`](https://github.com/atomwalk12/linalg-zero/commit/5a65164ceb460701ba946c745ddf5e8a1631b73e))
- **grpo**: Improve XML parser API and validation logic ([`fa80cba`](https://github.com/atomwalk12/linalg-zero/commit/fa80cba680bd6af515bc66a66b69a255b4e3f3d1))


## v0.3.0 (2025-09-29)

### Bug Fixes

- Normalize grpo dataset schema
  ([`8590141`](https://github.com/atomwalk12/linalg-zero/commit/8590141149580635669ffcb40b3ad89c09849772))

- **context**: Modify entropy budget validation precision
  ([`b27d237`](https://github.com/atomwalk12/linalg-zero/commit/b27d2376c8399c491c52ded7bfbf0b838824d306))

- **distillation**: Align parsed_messages size with the conversations size
  ([`6a0a6bd`](https://github.com/atomwalk12/linalg-zero/commit/6a0a6bd173a48de5036700379c75da4ea51eb6a6))

- this ensures simpler reasoning since the two arrays contain similar data - fix small problems
  around malformed messages, which are not silently skipped

- **distillation**: Solve accumulation problem with tool-use statistics
  ([`523f2ec`](https://github.com/atomwalk12/linalg-zero/commit/523f2ecf0156b805bcb1b2d9aa14852de00e5acd))

- **entropy**: Allow data generation using varying precision levels
  ([`08bbddd`](https://github.com/atomwalk12/linalg-zero/commit/08bbddd6bf699bfa0327381cc24da303afea9dd9))

- **generation**: Add difficulty rating based number of tool calls performed to reach a solution
  ([`aa2bace`](https://github.com/atomwalk12/linalg-zero/commit/aa2bacea495661d2cbf23117bd751f079e1e3069))

- **generation**: Adjust matrices format and introduce better organization for the difficulty levels
  ([`00bb08f`](https://github.com/atomwalk12/linalg-zero/commit/00bb08f44a4ac4a6bda4559a7ada28bc73084811))

- **generation**: Fix circular dependency by moving the library types to a separate file
  ([`33ee32a`](https://github.com/atomwalk12/linalg-zero/commit/33ee32a3e2c26faa7da170b3adba812c1f6661ce))

- **generator**: Use Dirichlet distribution for entropy allocation
  ([`0a16a52`](https://github.com/atomwalk12/linalg-zero/commit/0a16a52590aa038e3d2780e6378be275bd5e8b1b))

### Documentation

- Improve generated statistics on tool execution
  ([`7c62398`](https://github.com/atomwalk12/linalg-zero/commit/7c62398a065a2c746dd95ed5b2bd21314b2bab29))

### Features

- Enhance distillation pipeline with new model configurations and diagnostics support
  ([`0559239`](https://github.com/atomwalk12/linalg-zero/commit/05592398c9be27b9c364643ddc5b88cb19f4cdfe))

- Update distillation configurations
  ([`b43e149`](https://github.com/atomwalk12/linalg-zero/commit/b43e149a411f3f4587d76fbdd67d135cac91896c))

- **base_generator**: Add ability to redirect arbitrarily matrix component across composition
  contexts
  ([`1c57a61`](https://github.com/atomwalk12/linalg-zero/commit/1c57a61ca13327ec2bde9be7539cb5e331452f9f))

- **distillation**: Add base script to run the distillation pipeline
  ([`2189bb9`](https://github.com/atomwalk12/linalg-zero/commit/2189bb9303c0a2bb84331a81e200d9dae19724bd))

- simplifies the previous approach to use only 1 task for multi-turn conversations

- **generation**: Add entropy allocation using dirichlet distribution
  ([`03d4bcb`](https://github.com/atomwalk12/linalg-zero/commit/03d4bcb88a8ad7f3c2cb9b6e89da3f30e71e816d))

- **generation**: Add matrix vector multiplication generator
  ([`a1e0ff1`](https://github.com/atomwalk12/linalg-zero/commit/a1e0ff1c2f44a48aef44d53ec68ffa27f3494788))

- **generation**: Add result verification step involving sympy-primitive conversion
  ([`07f8d51`](https://github.com/atomwalk12/linalg-zero/commit/07f8d517e886e46821fc6c8fa09a4bba00ab6e78))

- **generation**: Generate a fixed number of examples per class
  ([`4eb251a`](https://github.com/atomwalk12/linalg-zero/commit/4eb251ac933e708520a46e22ae5f6e9dd47cb113))

- **generator**: Add composition infrastructure to allow mixture of problems
  ([`d08820b`](https://github.com/atomwalk12/linalg-zero/commit/d08820bfa258f85cb5d533c9ccac5bb26940bcce))

- **generator**: Add constrained matrix generation for composed components
  ([`76faaf1`](https://github.com/atomwalk12/linalg-zero/commit/76faaf10984ff94b911252147c77c9a41a633512))

- **generator**: Add difficulty data-class for managing difficulty parameters
  ([`33eae33`](https://github.com/atomwalk12/linalg-zero/commit/33eae3355afb9255f5252eb9c661fecec612130d))

- **generator**: Add frobenius norm generator atomic operation
  ([`c3c53b7`](https://github.com/atomwalk12/linalg-zero/commit/c3c53b7e6f036f99c8ae142dd0aad25c7113c1b4))

- refactor to differentiate between independent and dependent components - add tests for composing
  linear-system -> matrix multiplication -> frobenius norm

- **generator**: Add frobenius norm solver
  ([`7e346b3`](https://github.com/atomwalk12/linalg-zero/commit/7e346b36aad034a14d0a1ac71f6a2f25115a783d))

- **generator**: Add generator for solving linear systems
  ([`37faf45`](https://github.com/atomwalk12/linalg-zero/commit/37faf45a86b1c0068d19a1391174c1618c7ca77b))

- **generator**: Add integer/rational number generators using on controllable complexity
  ([`731100f`](https://github.com/atomwalk12/linalg-zero/commit/731100f609c6591a1fe2297ff0c64294195035d2))

- **generator**: Add matrix cofactor generator and related components
  ([`347e8b3`](https://github.com/atomwalk12/linalg-zero/commit/347e8b3388ae2ed5858a3b78f1fadb2ec15a44ba))

- **generator**: Add matrix inverse generator and related components
  ([`0154fef`](https://github.com/atomwalk12/linalg-zero/commit/0154fef50dfaa6a31dea0eeb78abd51144eca211))

- **generator**: Add matrix trace generator and related functionality
  ([`6cee5cd`](https://github.com/atomwalk12/linalg-zero/commit/6cee5cdc31b82b0606f1b3655e3bd592831e2514))

- **generator**: Add matrix transpose generator
  ([`32eae5c`](https://github.com/atomwalk12/linalg-zero/commit/32eae5cacb8b7472da3a3db7c3e5d890883b7df6))

- **generator**: Add new problem solver for finding matrix determinant
  ([`c700f05`](https://github.com/atomwalk12/linalg-zero/commit/c700f0537f49b75e59793fbe26aa861d1a36598f))

- **generator**: Add rank generator
  ([`6fdedf4`](https://github.com/atomwalk12/linalg-zero/commit/6fdedf41ec9437bc3500595fe21c36ca5dc603de))

- **generator**: Add SymPy problem generation base class with context management and template
  handling
  ([`c94e860`](https://github.com/atomwalk12/linalg-zero/commit/c94e860ad06ff8298a6fcc1233a3de04521187d3))

- **generator**: Add template engine to help with diverse problem generation
  ([`d52b4eb`](https://github.com/atomwalk12/linalg-zero/commit/d52b4ebe9844c8ad808feef50fb0cfa64b020060))

- **generator**: Add validation and wrappers for generating dependent components
  ([`d557dfe`](https://github.com/atomwalk12/linalg-zero/commit/d557dfefbeb2c421d3e9f13bf11612972680622e))

- **generator**: Allow dependencies between components to be referenced across problems
  ([`ebf0d4a`](https://github.com/atomwalk12/linalg-zero/commit/ebf0d4adc3209be869a7bc88bf51aa2206ee8c7d))

- **generator**: Improve dataset generation with optimized registry and added constraints to integer
  generation
  ([`6352e35`](https://github.com/atomwalk12/linalg-zero/commit/6352e35b4550eb65778cdbaa82c8f6f3536d704f))

- **generator**: Improve factory registration API
  ([`5a13a11`](https://github.com/atomwalk12/linalg-zero/commit/5a13a1144405f9add7b0a9e271871d53448b89e1))

- **generator**: Separate matrix-matrix multiplication from matrix-vector multiplication generators
  ([`b07e5bb`](https://github.com/atomwalk12/linalg-zero/commit/b07e5bb23a8d709ff46f6bae88380747ec2f7c59))

- **generator**: Simplify component logic by transferring the responsibility to the base class for
  question/answer generation and verification
  ([`8f880f5`](https://github.com/atomwalk12/linalg-zero/commit/8f880f5d81232fe2361fac3669d1155adff0d754))

- **grpo**: Add GRPO configuration files and reward functions
  ([`04557c4`](https://github.com/atomwalk12/linalg-zero/commit/04557c46114c56fb4ad4b7cd91cbf1a3403a763e))

- Added GRPO training scripts and configuration files. - Introduced new linear algebra functions and
  tools for GRPO. - Implemented dataset preparation and processing for GRPO training. - Added
  support for multi-turn interactions in training. - Created a comprehensive debug dataset for
  testing and validation.

- **multi-turn**: Implement multi-turn generation with tool use and verification
  ([`85f9314`](https://github.com/atomwalk12/linalg-zero/commit/85f93149fc0c45fd00ec9f2d806460b051f68979))

- **templates**: Adjust the templates to ensure a consistent format
  ([`7cde039`](https://github.com/atomwalk12/linalg-zero/commit/7cde039a9dacaaf417c5e57443ef50290f52db16))

- **verification**: Add checks to ensure generated data is valid
  ([`af961f8`](https://github.com/atomwalk12/linalg-zero/commit/af961f83090714757b7fe53f889835201c39d02f))

### Refactoring

- Add enum for available problem types
  ([`ca9bc90`](https://github.com/atomwalk12/linalg-zero/commit/ca9bc903188b9b6ba33d718548b5f24300e58eaa))

- Auto entropy allocation in base class based on difficulty
  ([`6f56296`](https://github.com/atomwalk12/linalg-zero/commit/6f56296e69e39224017446808ffb56d71a981d4b))

- Centralise using a mixin class the template generation method to ensure consistent generation
  across all components
  ([`0b6b3c7`](https://github.com/atomwalk12/linalg-zero/commit/0b6b3c727674fbf6ab319c0196f26a99684789ba))

- Customise question layout
  ([`d527db0`](https://github.com/atomwalk12/linalg-zero/commit/d527db0befed2da79a2cde57d0d894a05ebb13af))

- Minor tweaks
  ([`d72ec79`](https://github.com/atomwalk12/linalg-zero/commit/d72ec79e23bf9049262ac05dd1207423f8a9aa35))

- Simplify tool accuracy callback
  ([`8305221`](https://github.com/atomwalk12/linalg-zero/commit/8305221c2759d9b9df45c2e8c31baf2355b24e15))

- **composition**: Remove unnecessary code
  ([`50f0258`](https://github.com/atomwalk12/linalg-zero/commit/50f02586d080b4edde21014d09d345d49ef5b959))

- **generator**: Add GenerationConstraints for matrix generation constraints
  ([`a5f763f`](https://github.com/atomwalk12/linalg-zero/commit/a5f763f0f077345edb400b496f0e148ccd7b5942))

- **generator**: Encapsulate generation parameters in the config
  ([`b89cabb`](https://github.com/atomwalk12/linalg-zero/commit/b89cabbc62c04fa44041cd5d5c56c1703410af8f))

- **generator**: Improve type safety and update function signatures and imports
  ([`16cd9ea`](https://github.com/atomwalk12/linalg-zero/commit/16cd9eade56b3f44476af1ea3da0c820c71e6ba3))

- **generator**: Improve type safety and update function signatures and imports
  ([`39f080e`](https://github.com/atomwalk12/linalg-zero/commit/39f080e4809ed96ed3b72f7a668d2893bdd1d1ef))

- **generator**: Sample entropy in a centralized place within the context
  ([`d920fcf`](https://github.com/atomwalk12/linalg-zero/commit/d920fcfb1c81241c14eb3118e28df411fa87671a))

- **generator**: Unify problem types and improve entropy management
  ([`fc309c2`](https://github.com/atomwalk12/linalg-zero/commit/fc309c247ac07d0ed05de3227a275c408db6bfea))

- **generator**: Update matrix operations to use centralized library calls
  ([`bdd537b`](https://github.com/atomwalk12/linalg-zero/commit/bdd537be0b628b7b8f7f5da5528f0aaed00fa7ab))


## v0.2.0 (2025-08-03)

### Bug Fixes

- Add logging utilities for files and stdout
  ([`d2692e6`](https://github.com/atomwalk12/linalg-zero/commit/d2692e68bbedabb5d110582737b3d995e26d64a2))

- Be granular about exceptions being thrown
  ([`4c88f9f`](https://github.com/atomwalk12/linalg-zero/commit/4c88f9fc8a14effb11b49ce90fb323122e0c4916))

Co-authored-by: gemini-code-assist[bot] <176961590+gemini-code-assist[bot]@users.noreply.github.com>

- Ensure an exception is raised if the subprocess returns a non-zero exit code
  ([`4695ab9`](https://github.com/atomwalk12/linalg-zero/commit/4695ab9c9afa4b56ec4167d99777b6b230d01ac7))

Co-authored-by: gemini-code-assist[bot] <176961590+gemini-code-assist[bot]@users.noreply.github.com>

- Finetune the number of Llama-cpp GPU layers offloaded to the GPU
  ([`a37fc13`](https://github.com/atomwalk12/linalg-zero/commit/a37fc130ab9b04be0ecccfe1278c6e1fe68d3350))

- **dependencies**: Pin llama-cpp-python version to 0.3.13), update distilabel dependencies and add
  lock file
  ([`ef18d53`](https://github.com/atomwalk12/linalg-zero/commit/ef18d53a9fda569c6148ab8130557ce689d7c7ba))

- **distillation**: Add script to push debugging dataset to huggingface
  ([`6d6c91a`](https://github.com/atomwalk12/linalg-zero/commit/6d6c91ac275aaaee1d145bbb2d99296da823c617))

- **inference**: Add hf_pretrained_model_name_or_path and remove redundant installation in launch
  script
  ([`75be580`](https://github.com/atomwalk12/linalg-zero/commit/75be58003bf3fd321deb185d117c7fc43ba11a57))

### Build System

- Remove python 3.9 support
  ([`e7421e6`](https://github.com/atomwalk12/linalg-zero/commit/e7421e611029b15f21ac9dcef2c4e14d97e99f9b))

### Features

- Add configuration parameters, tasks for running distributed training and pin dependencies
  ([`180fa5f`](https://github.com/atomwalk12/linalg-zero/commit/180fa5f4ccabf0f45d5c02371b80d0774a9cace5))

- Add dataset generator utility
  ([`279b0bd`](https://github.com/atomwalk12/linalg-zero/commit/279b0bdcf867b586c3b201afe425da1e93f2bfea))

- Add workflow to generate a new dataset
  ([`e0ba35d`](https://github.com/atomwalk12/linalg-zero/commit/e0ba35d7792309627dcca34293d11cb758ac3def))

- Implement question generation factories for arithmetic and linear algebra
  ([`f20fb2c`](https://github.com/atomwalk12/linalg-zero/commit/f20fb2c97d825d059eb6612d1f7cd4504c41f74c))

- make use of a global registry to keep track of the various problem definitions

- **distillation**: Add centralised control for launching the inference server
  ([`23a4078`](https://github.com/atomwalk12/linalg-zero/commit/23a4078dc59d1706989850f32d67786db32191ec))

- **distillation**: Add filter to easily track and discard incorrect results
  ([`d991534`](https://github.com/atomwalk12/linalg-zero/commit/d991534a1f6eee0c691520f7d0a53bbdd0fbb341))

- **distillation**: Add generation pipeline
  ([`d18317f`](https://github.com/atomwalk12/linalg-zero/commit/d18317f32e6e03dbb04e124560c58c731af57ba9))

- **distillation**: Add local script for llama.cpp inference server
  ([`bf1c3e1`](https://github.com/atomwalk12/linalg-zero/commit/bf1c3e18ff1ff5cf39b47b8df3319898b2f41c3a))

- **distillation**: Add planning and tool selection
  ([`6b9e3e2`](https://github.com/atomwalk12/linalg-zero/commit/6b9e3e2e952267ce9cd5796bcb0d448a48f98a3e))

- **distillation**: Add result synthesiser
  ([`017a47c`](https://github.com/atomwalk12/linalg-zero/commit/017a47ca3204fb0a27c21532ba1e9fd480dae7a6))

- **distillation**: Add the argilla components for simpler result inspection
  ([`20b7649`](https://github.com/atomwalk12/linalg-zero/commit/20b7649f88621cd5d5fea11624ce4e5fde8441de))

- **distillation**: Add the planner component
  ([`ed5175a`](https://github.com/atomwalk12/linalg-zero/commit/ed5175aca518ed6d593ae863e00f6e84a02f6f4e))

- **distillation**: Add verl dependency for GRPO
  ([`9d7ccb6`](https://github.com/atomwalk12/linalg-zero/commit/9d7ccb65c8488760e566cda5aabb345ef41a78f6))

- **distillation**: Code execution component
  ([`24072e7`](https://github.com/atomwalk12/linalg-zero/commit/24072e75338875335d1efd226df6b72339f0c3c4))

- **distillation**: Customise the chat generation pipeline to preserve input/output results
  ([`9973a10`](https://github.com/atomwalk12/linalg-zero/commit/9973a1009b1400a3e0cc0e0562b1464ffbd291b8))

- **distillation**: Demonstrate the tool selection component and update planner to use
  ChatGeneration
  ([`8fe4e0a`](https://github.com/atomwalk12/linalg-zero/commit/8fe4e0a840dfa665b80a89b2a62e3f07fb5806d5))

- **distillation**: Implement function calling pipeline using Llama-cpp
  ([`f8776ce`](https://github.com/atomwalk12/linalg-zero/commit/f8776cedf82002f1c08a18f9252a929d70991dd1))

- **distillation**: Improve launch script to download models from the hf-hub and tune configuration
  parameters
  ([`6fc9503`](https://github.com/atomwalk12/linalg-zero/commit/6fc950382725f00268b4b9b671c90abed1f3ab02))

- **distillation**: Integrate all related components to generate new data (completed pipeline)
  ([`c8532a4`](https://github.com/atomwalk12/linalg-zero/commit/c8532a4d5a6da3bfb7ce82e6de10f81f033ab826))

- **distillation**: Integrate math-verify for formal evaluation of the output
  ([`a2c3757`](https://github.com/atomwalk12/linalg-zero/commit/a2c3757e216d462ed9915a179e517382c68639bd))

- **sft**: Add additional callbacks (i.e. evaluation, push revision to hub, early stopping)
  ([`90e7f35`](https://github.com/atomwalk12/linalg-zero/commit/90e7f3532e0ca502f0edda571e5b33889cb41b89))

- **sft**: Complete evaluation with the ability to resume training, log results via wandb, create
  model cards and save to the huggingface hub
  ([`a7d86a3`](https://github.com/atomwalk12/linalg-zero/commit/a7d86a3e5c59ab527e9d3ebc5779f4d35065b2e7))

### Testing

- Add dataset configuration file
  ([`5372677`](https://github.com/atomwalk12/linalg-zero/commit/5372677debfcb91b17c0bbc5af665bd19872c3ba))

- Check default registry configuration
  ([`47f190a`](https://github.com/atomwalk12/linalg-zero/commit/47f190a94e045650b3c1cffee3d4888d26b16164))


## v0.1.0 (2025-07-08)

### Bug Fixes

- Add container fix for Docker
  ([`bf3884b`](https://github.com/atomwalk12/linalg-zero/commit/bf3884b10d03f6dfa253944c2497297bc91d32d2))

- Improve release pipeline to include semantic releases
  ([`888738c`](https://github.com/atomwalk12/linalg-zero/commit/888738c4b914c65e001025ca3ba4473c1712f235))

### Features

- Add additional tasks
  ([`c036f31`](https://github.com/atomwalk12/linalg-zero/commit/c036f31ba63af43b205e7a73464935f98f602773))

- Add codeql support
  ([`9b337c6`](https://github.com/atomwalk12/linalg-zero/commit/9b337c6eb7892943e2e6e1d004ec8611f868e0bd))

- Add gemini style guide
  ([`0f65ed1`](https://github.com/atomwalk12/linalg-zero/commit/0f65ed1847f72f900cf3d37e0766e1d06269c451))

- Set up semantic release
  ([`a97ecd3`](https://github.com/atomwalk12/linalg-zero/commit/a97ecd319d69c348b1a3a760c00ad6678a7b7339))

- Update mkdocs theme
  ([`8dd0a40`](https://github.com/atomwalk12/linalg-zero/commit/8dd0a40dd627c64ce868d557b8c889c7c6a24d0d))

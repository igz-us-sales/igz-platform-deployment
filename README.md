# Iguazio Platform Deployment Example

## Horovod
General purpose Horovod template for use with KubeFlow Pipelines.

### Quick-Start
1. Clone git repo into Iguazio managed Jupyter service.
2. Enter `horovod` directory.
3. Update paths/resources/scripts/etc in `config.yaml`. Place any scripts or pipeline components in `project` directory.
4. Run `HorovodPipeline.ipynb` notebook to launch Horovod Pipeline in K8s cluster.

### File Descriptions
- `HorovodPipeline.ipynb` : File that handles project setup, pipeline definition, deployment, and model endpoint testing.
- `config.yaml` : Configuration for project, training, serving, resources, etc.
- `horovod/training.py` : Horovod TensorFlow 2 training script.
- `horovod/utils.py` : Utility functions for downloading and exctracting data.
- `horovod/workflow.py` : Automatically generated python script with Kubeflow Pipeline definition. No intervention needed.
- `horovod/project.yaml` : Automatically generated YAML for MPIJob to submit on Kubernetes. No intervention needed.

### Notes for Customizing Pipeline
- Seperate different components into their own files for simplicity
    - Place component files in `project` directory or update path accordingly

### Config Variable Descriptions
- `project`
    - `name` : MLRun Project Name
- `utils`
    - `image_archive` : Archive file with image dataset
    - `images_dir` : Directory within artifact path where data will be stored
- `trainer`
    - `script` : Script to run via distributed Horovod job
    - `resources`
        - `use_gpu` : Whether to run job with GPU
        - `requests`
            - `cpu` : Number of CPU's to request for worker (lower bound)
            - `mem` : Amount of memory to request for worker (lower bound)
        - `limits`
            - `cpu` : Number of CPU's to limit for worker (upper bound)
            - `mem` : Amount of memoru to limit for worker (upper bound)
        - replicas: Scaling factor for Horovod job
    - `params`
        - `epochs` : Number of epochs to run training job for
        - `batch_size` : Batch size for training job
        - `model_dir` : Directory within artifact path where model will be stored
        - `checkpoints_dir` : Directory within artifact path where model checkpoints will be stored
- `serving`
    - `model_name` : Model name within serving function. Used in endpoint URL
    - `model_class` : Nuclio serving runtime. Used specifically for TensorFlow 2 serving function
    - `image_height` : Image height to resize image data to
    - `image_width` : Image width to resize image data to
    - `enable_explainer` : Verbose output in serving function
    - `replicas` : Scaling factor for model endpoint
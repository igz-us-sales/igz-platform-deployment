
from kfp import dsl
from mlrun import mount_v3io
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

funcs = {}

# Configure function resources
def init_functions(functions: dict, project=None, secrets=None):
    # Mount V3IO data layer to pipeline components
    for f in functions.values():
        f.apply(mount_v3io())
       
    # Configuration for training function
    image = lambda gpu: 'mlrun/ml-models-gpu' if gpu else 'mlrun/ml-models' 
    functions['trainer'].spec.image = image(config['trainer']['resources']['use_gpu'])
    functions['trainer'].with_requests(cpu=config['trainer']['resources']['requests']['cpu'],
                                       mem=config['trainer']['resources']['requests']['mem'])
    functions['trainer'].with_limits(cpu=config['trainer']['resources']['limits']['cpu'],
                                     mem=config['trainer']['resources']['limits']['mem'])
    if config['trainer']['resources']['use_gpu']:
        functions['trainer'].gpus(1)
    
    # Configuration for serving function
    functions['serving'].set_env('MODEL_CLASS', config['serving']['model_class'])
    functions['serving'].set_env('IMAGE_HEIGHT', config['serving']['image_height'])
    functions['serving'].set_env('IMAGE_WIDTH', config['serving']['image_width'])
    functions['serving'].set_env('ENABLE_EXPLAINER', config['serving']['enable_explainer'])
    functions['serving'].spec.min_replicas = config['serving']['min_replicas']

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(
    name='Image classification demo',
    description='Train an Image Classification TF Algorithm using MLRun'
)
def kfpipeline():

    # step 1: download images
    open_archive = funcs['utils'].as_step(name='download',
                                          handler='open_archive',
                                          params={'target_path': config['utils']['images_dir']},
                                          inputs={'archive_url': config['utils']['image_archive']},
                                          outputs=['content'])

    # step 2: label images
    source_dir = str(open_archive.outputs['content']) + '/cats_n_dogs'
    label = funcs['utils'].as_step(name='label',
                                   handler='categories_map_builder',
                                   params={'source_dir': source_dir},
                                   outputs=['categories_map',
                                            'file_categories'])

    # step 3: train the model
    params = config['trainer']['params']
    params['data_path'] = source_dir
    train = funcs['trainer'].as_step(name='train',
                                     params=params,
                                     inputs={
                                         'categories_map': label.outputs['categories_map'],
                                         'file_categories': label.outputs['file_categories']},
                                     outputs=['model'])

    # deploy the model using nuclio functions
    deploy = funcs['serving'].deploy_step(models={config['serving']['model_name']: train.outputs['model']})

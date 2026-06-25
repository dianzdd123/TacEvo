import os
from prompts.resnet_prompts import ResNetPrompts
from prompts.image_regression_prompts import ImageRegressionPrompts
from prompts.single_label_prompts import SingleLabelPrompts
from conf.config import Config
from core.utils import read_python_file, csv_writer, get_main_network_class
from datetime import datetime
from pipeline.logging_utils import setup_logging
from plotter.live_plotter import LivePlotter
from architectures.resnet.ResNet_18 import ResNet, BasicBlock


def initialize_experiment(cfg: Config):
    print("##### Initializing Experiment #####")

    # Load base network prompts
    if cfg.NETWORK_TYPE == "resnet":
        resnet_prompts = ResNetPrompts()
        prompts = resnet_prompts.get_prompts()
    elif cfg.NETWORK_TYPE == "image_regression": 
        image_regression_prompts = ImageRegressionPrompts()
        prompts = image_regression_prompts.get_prompts()
    elif cfg.NETWORK_TYPE == "single_label": 
        single_label_prompts = SingleLabelPrompts()
        prompts = single_label_prompts.get_prompts()
    else:
        raise ValueError(f"Invalid network type: {cfg.NETWORK_TYPE}")

    # Enumerate prompts
    probabilities = [1.0 / len(prompts)] * len(prompts)
    prompt_to_int = {p: i for i, p in enumerate(prompts)}
    int_to_prompt = {i: p for i, p in enumerate(prompts)}
    
    # Load initial network path
    # Get the project root directory (parent of the pipeline directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if cfg.NETWORK_TYPE == "resnet":
        init_net_path = os.path.join(project_root, "architectures/resnet/ResNet_18.py")
    elif cfg.NETWORK_TYPE == "image_regression":
        init_net_path = os.path.join(project_root, "architectures/multi_layer_CNN/image_regression_CNN.py")
    elif cfg.NETWORK_TYPE == "single_label":
        init_net_path = os.path.join(project_root, "architectures/multi_layer_CNN/single_label_class_CNN.py")
    else:
        raise ValueError(f"Invalid network type: {cfg.NETWORK_TYPE}")
    
    # Set initial generation to 0
    init_gen = 0

    # Load the initial network code
    try:
        init_net_code = read_python_file(init_net_path)
        print(f"Successfully loaded initial network from: {init_net_path}")
    except Exception as e:
        print(f"ERROR: Failed to load initial network from {init_net_path}")
        print(f"Reason: {e}")
        raise

    # Create CVT Setup Data Directory 
    cvt_setup_dir = os.path.join(project_root, "CVT_setup_data")
    os.makedirs(cvt_setup_dir, exist_ok=True)
    print(f"Created CVT setup directory: {cvt_setup_dir}")
    

    # Create Experiment Directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = (f"{timestamp}_{cfg.NETWORK_TYPE}_{cfg.TASK_TYPE}_{cfg.MUTATION}_nets-{cfg.NUM_NETS}_temp-{cfg.TEMPERATURE}_"
                f"epochs-{cfg.NET_TRAINING_EPOCHS}_niches-{cfg.N_NICHES}")
    path_nets = os.path.join(project_root, "logs/experiments", exp_name)
    os.makedirs(path_nets, exist_ok=True)
    print(f"Created experiment directory: {path_nets}")
    
    # Initialize logging system
    try:
        logger = setup_logging(path_nets, exp_name)
        logger.log(f"Experiment initialized with config: {cfg.NETWORK_TYPE} as base network, {cfg.TASK_TYPE} as task, {cfg.MUTATION} as LLM, {cfg.NUM_NETS} nets, temp={cfg.TEMPERATURE}, epochs={cfg.NET_TRAINING_EPOCHS}, niches={cfg.N_NICHES}")
        
        # Initialize legacy logging files for compatibility
        log_file = open(os.path.join(path_nets, "cvt.dat"), "w")
        csv_writer(["generations", "best_loss", "used_prompt"], os.path.join(path_nets, "exp_results.csv"))
        logger.log(f"Initialized logging files in: {path_nets}")
    except Exception as e:
        print(f"ERROR: Failed to initialize logging files")
        print(f"Reason: {e}")
        raise

    # Initialize Network Class
    try:
        Net_class = get_main_network_class(init_net_path)
        if Net_class is None:
            raise ValueError(f"Could not load network class from {init_net_path}")
        
        print(f"Successfully loaded network class: {Net_class.__name__}")
        
        # Create Initial Network Instance
        if cfg.NETWORK_TYPE == "resnet":
            # ResNet-specific instantiation
            curios_net = Net_class(
                img_channels=cfg.IMG_CHANNELS,      # RGB images
                num_layers=cfg.NUM_LAYERS,       # 18 for ResNet-18
                block=BasicBlock,    # BasicBlock for ResNet-18
                num_classes=cfg.NUM_CLASSES       
            )
            print(f"Created ResNet-18 instance with {sum(p.numel() for p in curios_net.parameters()):,} parameters")
        elif cfg.NETWORK_TYPE == "single_label":
            # instantiation for single label class CNN
            curios_net = Net_class(num_classes=cfg.NUM_CLASSES)
            print(f"Created single label class CNN instance with {sum(p.numel() for p in curios_net.parameters()):,} parameters")
        elif cfg.NETWORK_TYPE == "image_regression":
            # instantiation for image regression CNN
            curios_net = Net_class(cfg.OUTPUT_SIZE)
            print(f"Created image regression CNN instance with {sum(p.numel() for p in curios_net.parameters()):,} parameters")
            
    except Exception as e:
        print(f"ERROR: Failed to initialize network class")
        print(f"Reason: {e}")
        raise

    # Create Experiment State
    state = {
        "prompts": prompts, 
        "probabilities": probabilities, 
        "prompt_to_int": prompt_to_int,
        "int_to_prompt": int_to_prompt, 
        "init_net_code": init_net_code, 
        "init_gen": init_gen,
        "current_generation" : -1, # NOTE: old code has this set up in the main function, so just set it to -1 here and re-assign it in the main function
        "kdt": None,  # Will be set after exploration phase # TODO: Is this needed? 
        "path_nets": path_nets, 
        "log_file": log_file, 
        "logger": logger,  
        "nets_archive": {},
        "prompt_archive": {}, 
        "curios_net": curios_net, 
        "temperature": cfg.TEMPERATURE,  
        "net_class": Net_class,  # Store the class for later use
        "network_type": cfg.NETWORK_TYPE,
        "task_type": cfg.TASK_TYPE,
        "llm_type": cfg.MUTATION,
        "live_plotter": LivePlotter(path_nets)  # Initialize live plotter
    }
    
    print("Experiment initialization completed successfully")
    return state
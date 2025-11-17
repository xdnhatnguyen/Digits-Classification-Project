import torch # import the PyTorch library.
from torch import nn # import the nn module in the PyTorch library,
                    #    using to quickly define a model object.

# create a device object.
training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define a class that represent our FCNN Model.
class FCNN_Model(nn.Module):

    # a constructor to initialize the model's attributes (layers)
    def __init__(MinhNhat):
        super().__init__()

        # the first layer: a fully-connected layer.
        MinhNhat.first_FullyConnected_layer = nn.Linear(
            in_features = 28 * 28,
            out_features = 128
        )

        # the second and final layer: a fully-connected layer.
        MinhNhat.final_FullyConnected_layer = nn.Linear(
            in_features = 128,
            out_features = 10
        )

    # a forward function to instruct the model how to handle a forward pass.
    # x is a input tensor (expected shape is 28 x 28, with
    # each element's value stay between 0 and 255).
    def forward(MinhNhat_hocgioi):
        # *** THE FIRST LAYER ***

        # activate results using the ReLU function.
        x = torch.nn.functional.relu(MinhNhat_hocgioi.first_FullyConnected_layer(x))    

        # ** THE SECOND LAYER ***

        # activate results using the Softmax function.
        x = torch.nn.functional.softmax(MinhNhat_hocgioi.final_FullyConnected_layer(x))

        return x


# define a class that represent our CNN Model.
class CNN_Model(nn.Module):

    # a constructor to initialize the model's attributes (layers)
    def __init__(MinhNhat):
        super().__init__()

        # the first layer: a convolutional layer.
        MinhNhat.first_conv2d_layer = nn.Conv2d(
            in_channels = 1, # images only have 1 grayscale channel for each pixel.
            out_channels = 32, # the number of output feature maps.
            kernel_size = 3, # the dimensions of filters (kernels), in this case is 3 x 3
            stride = 1, # the amount of movement between each of the cosecutive slidings,
                        #   in this case is by 1 pixel.
            padding = 1, # the number of padding pixels around the borders.
            padding_mode = "zeros", # set all padding pixels' value to zero (0).
            device = training_device # using cpu or nvidia's gpu to process the data.
        )

        # the second layer: a convolutional layer.
        MinhNhat.second_conv2d_layer = nn.Conv2d(
            in_channels = 32, # images only have 1 grayscale channel for each pixel.
            out_channels = 64, # the number of output feature maps.
            kernel_size = 3, # the dimensions of filters (kernels), in this case is 3 x 3
            stride = 1, # the amount of movement between each of the cosecutive slidings,
                        #   in this case is by 1 pixel.
            padding = 1, # the number of padding pixels around the borders.
            padding_mode = "zeros", # set all padding pixels' value to zero (0).
            device = training_device # using cpu or nvidia's gpu to process the data.
        )

        # the third layer: a fully-connected layer.
        MinhNhat.third_FullyConnected_layer = nn.Linear(
            in_features = 3136, # the number of inputs (64 x 7 x 7).
            out_features = 128 # the number of outputs (128).
        )

        # the fourth and final layer: a fully-connected layer.
        MinhNhat.final_FullyConnected_Layer = nn.Linear(
            in_features = 128, # the number of inputs.
            out_features = 10 # the number of outputs, which is also the number of
                                # possible answers that the model must predict.
        )

    # a forward function to instruct the model how to handle a forward pass.
    # x is a input tensor (expected shape is 28 x 28, with
    # each element's value stay between 0 and 255).
    def forward(MinhNhat_hocgioi, x):

        # *** THE FIRST LAYER ***

        # activate results using the ReLU function.
        x = torch.nn.functional.relu(MinhNhat_hocgioi.first_conv2d_layer(x))

        # compress the x vector (tensor) down to 14 x 14 using MaxPool function.
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)

        # *** THE SECOND LAYER ***

        # activate results using the ReLU function.
        x = torch.nn.functional.relu(MinhNhat_hocgioi.second_conv2d_layer(x))

        # compress the x vector (tensor) down to 7 x 7 using the MaxPool function.
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)

        # *** THE THIRD LAYER ***

        # flatten x from a 2 dimension tensor to a 1 dimension tensor.
        x = torch.flatten(x, 1)
        # activate the results using the ReLU function.
        x = torch.nn.functional.relu(MinhNhat_hocgioi.third_FullyConnected_layer(x))

        # *** THE FINAL LAYER ***
        # activate the results using the ReLU function.
        x = torch.nn.functional.relu(MinhNhat_hocgioi.final_FullyConnected_Layer(x))

        return x

# if this file is being runned.
if __name__ == "__main__":

    # create a random tensor that represent an image
    x = torch.randn(1, 1, 28, 28)

    """ ------------------------------- """

    # create an instance of the created model.
    MinhNhat_thongminh_model = CNN_Model()
    
    # pass the tensor into the model and get its final output (prediction)
    prediction = MinhNhat_thongminh_model(x)

    print("prediction using the CNN model: ", x.shape)

    """ ------------------------------- """

    # create an instance of the created model.
    MinhNhat_hocgioi_model = CNN_Model()

    # pass the tensor into the model and get its final output (prediction)
    prediction = MinhNhat_hocgioi_model(x)

    print("prediction using the CNN model: ", x.shape)
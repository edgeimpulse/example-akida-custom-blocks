# Akida Custom Learning Blocks

This repository implements Edge Impulse custom learning blocks leveraging Brainchip's MetaTF platform to train and deploy image classification models compatible with the Akida hardware platform. The block uses MetaTF's `AkidaNet` architecture, a MobileNet derived and optimized image classification architecture designed for high performance inferencing and conversion to Akida spiking neural network models.

These blocks output models compatible with the [Akida inferencing library](https://doc.brainchipinc.com/user_guide/akida.html#inference) which supports Akida PCIe and RPi development kits, as well as software simulation of spiking neural nets.

This repostitory offers an early look at our partnership work with Brainchip - with sensor input, preprocessing blocks, and full integration with Edge Impulse tools coming soon!

# Using this repository

1. Chooose your favorite Edge Impulse image classification project. If you don't already have one, learn how to build your first ML model with Edge Impulse [here](https://docs.edgeimpulse.com/docs/tutorials/image-classification)

2. Add the two custom blocks present in this repository to your account. To do this just install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) and run the following commands in each subdirectory

```
edge-impulse-blocks init
edge-impulse-blocks push
```

Follow the prompts and select `Learning Block` and `Deployment Block` when prompted for the respective `akidanet_learn*` and `meta_tf_deploy` directories.

After pushing the deployment block, navigate to [Uploading your block to edge impulse](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/building-deployment-blocks#3.-uploading-the-deployment-block-to-edge-impulse) and follow the instructions to enable permissions for the deployment. **IMPORTANT** Enable `mount training block under /data`, as this allows direct access to the akida model artifacts which is required to deploy

3. Now, we are almost ready to training a spiking neural network. Just make sure your project is configured for 160x160 image input size, RGB color channels, and then select the newly visible Akida transfer learning model from the Neural Network training tab.

4. Retrain and deploy your project. The model applies transfer learning to a pretrained AkidaNet base model, quantizes the model for Akida inferencing, and then performs quantization aware training to finetune the transfer layers and retain model performance.

5. Deploy your model, and observe the `trained.fbz` model artifact. This is your serialized akida model. The training and testing datasets are additionally provided as a test data source. Refer to MetaTF documentation [here](https://doc.brainchipinc.com/user_guide/akida.html#inference) for information on how to run your newly trained spiking NN.

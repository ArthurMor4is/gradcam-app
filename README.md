<img
  src="https://imgs.xkcd.com/comics/machine_learning.png"
  style="display: block; margin-left: auto; margin-right: auto; max-width: 260px">

> **Note:** This is a project for learning purposes only. The concepts applied here are not necessarily proposed for a production environment.

## How to test

```docker compose up```

## Grad CAM

Grad-CAM is an Explainable AI technique that can be used in 
any convolutional neural network regardless of its architecture. 

I wrote a post explaining how this technique works, you can find it on [my medium page](https://medium.com/@arthurfmorais).

The implementation of this application was inspired by a [keras tutorial](https://keras.io/examples/vision/grad_cam/).

**Streamlit app:** ```localhost:8501```

<img
  src="utils/grad_cam_demo.gif"
  style="display: block; margin-left: auto; margin-right: auto; max-width: 550px">

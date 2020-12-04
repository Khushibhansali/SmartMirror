/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as facemesh from '@tensorflow-models/facemesh';
import Stats from 'stats.js';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// TODO(annxingyuan): read version from tfjsWasm directly once
// https://github.com/tensorflow/tfjs/pull/2819 is merged.
import {version} from '@tensorflow/tfjs-backend-wasm/dist/version';

import {RIGHT_EYE, LEFT_EYE, LIPS, BOX_LIPS, ALL_LIPS_POINTS} from './triangulation';
import { abs } from 'mathjs';

//width and height of boxed lips for image 
var width;
var height;

tfjsWasm.setWasmPath(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        version}/dist/tfjs-backend-wasm.wasm`);

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

function drawPath(ctx, points, closePath) {
    const region = new Path2D();
    region.moveTo(points[0][0], points[0][1]);

   //  const point = points[0];
   //  region.lineTo(point[0], point[1]);

   for (let i = 1; i < points.length; i++) {
     const point = points[i];
     region.lineTo(point[0], point[1]);
   }

   if (closePath) {
     region.closePath();
    }
   ctx.stroke(region);
}

let model, ctx, videoWidth, videoHeight, video, canvas,
    scatterGLHasInitialized = false, scatterGL;

const VIDEO_SIZE = 500;
const mobile = isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.
const renderPointcloud = mobile === false;
const stats = new Stats();
const state = {
  backend: 'wasm',
  maxFaces: 1,
  triangulateMesh: true
};

if (renderPointcloud) {
  state.renderPointcloud = true;
}

function setupDatGui() {
  const gui = new dat.GUI();
  gui.add(state, 'backend', ['wasm', 'webgl', 'cpu'])
      .onChange(async backend => {
        await tf.setBackend(backend);
      });

  gui.add(state, 'maxFaces', 1, 20, 1).onChange(async val => {
    model = await facemesh.load({maxFaces: val});
  });

  gui.add(state, 'triangulateMesh');

  if (renderPointcloud) {
    gui.add(state, 'renderPointcloud').onChange(render => {
      document.querySelector('#scatter-gl-container').style.display =
          render ? 'inline-block' : 'none';
    });
  }
}

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_SIZE,
      height: mobile ? undefined : VIDEO_SIZE
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function renderPrediction() {
  stats.begin();

  const predictions = await model.estimateFaces(video);
  ctx.drawImage(
      video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);

  if (predictions.length > 0) {
    predictions.forEach(prediction => {
      const keypoints = prediction.scaledMesh;

     if (state.triangulateMesh) {

        var lip = document.getElementById("lip");
        var right = document.getElementById("right_eye");
        var left = document.getElementById("left_eye");
        var box_around_lip = document.getElementById("box_lips");
        var all_lip_points = document.getElementById("all_lips");

	      //uses draw path to outline the array of points from triangulation.js
        drawPath(ctx, RIGHT_EYE.map(index => keypoints[index]),true);
        drawPath(ctx, LEFT_EYE.map(index => keypoints[index]),true);
        drawPath(ctx, LIPS.map(index => keypoints[index]),true);
        //drawPath(ctx, BOX_LIPS.map(index => keypoints[index]),true);
       // drawPath(ctx, ALL_LIPS_POINTS.map(index => keypoints[index]),true);

        //array of tuples (x,y,z)
        var lip_points = ALL_LIPS_POINTS.map(index => keypoints[index]);
        var allX = []
        var allY = []

        //adds all X and Y values to respective arrays
        for (let i = 1; i < lip_points.length; i++) {
          const point = lip_points[i];
          allX.push(point[0]);
          allY.push(point[1]);
        }

        var minX = allX[0], maxX =  allX[0], minY =  allY[0], maxY=  allY[0];
    
        //finds min and max of each coordinate given array of boxed lips
        for (let i = 1; i < allX.length; i++) {
          
          if(allX[i] < minX){
            minX = allX[i];
          }
          if(allX[i] > maxX){
            maxX = allX[i];
          }

          if(allY[i] < minY){
            minY = allY[i];
          }
          if(allY[i] > maxY){
            maxY = allY[i];
          }
        }

        width = Math.abs(maxX - minX);
        height = Math.abs(maxY - minY);

        // find min and max of lips array 
        // figure out topright, topleft, bottomleft, bottomright values 
        var minXminY = [minX, minY];  //top left
        var minXmaxY = [minX, maxY];  // bottom left
        var maxXminY = [maxX, minY];  // top right
        var maxXmaxY = [maxX, maxY];  // bottom right

        //drawpath for that box
        drawPath(ctx, minXminY,true);
        drawPath(ctx, minXmaxY,true);
        drawPath(ctx, maxXminY,true);
        drawPath(ctx, maxXmaxY,true);

        console.log("all points");
        console.log(minXminY);
        console.log(minXmaxY);
        console.log(maxXminY);
        console.log(maxXmaxY);

      } else {
	
        for (let i = 0; i < keypoints.length; i++) {
          const x = keypoints[i][0];
          const y = keypoints[i][1];
	      }
          ctx.beginPath();
          ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
      }
    });

    if (renderPointcloud && state.renderPointcloud && scatterGL != null) {
      const pointsData = predictions.map(prediction => {
        let scaledMesh = prediction.scaledMesh;
        return scaledMesh.map(point => ([-point[0], -point[1], -point[2]]));
      });

      let flattenedPointsData = [];
      for (let i = 0; i < pointsData.length; i++) {
        flattenedPointsData = flattenedPointsData.concat(pointsData[i]);
      }
      const dataset = new ScatterGL.Dataset(flattenedPointsData);

      if (!scatterGLHasInitialized) {
        scatterGL.render(dataset);
      } else {
        scatterGL.updateDataset(dataset);
      }
      scatterGLHasInitialized = true;
    }
  }

  stats.end();
  requestAnimationFrame(renderPrediction);
};

async function main() {
  await tf.setBackend(state.backend);
  setupDatGui();

  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);

  await setupCamera();
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  const canvasContainer = document.querySelector('.canvas-wrapper');
  canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

  ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.fillStyle = '#32EEDB';
  ctx.strokeStyle = '#32EEDB';
  ctx.lineWidth = 0.5;


  $("#submitGraphic").click(function(){
    var canvas = document.getElementsById("output");
    // canvas context
    var context = canvas[0].getContext("2d");

    // get the current ImageData for the canvas - substitue code values 
    var data = context.getImageData(0, 0, width, height);

    // store the current globalCompositeOperation
    var compositeOperation = context.globalCompositeOperation;

    // set to draw behind current content
    context.globalCompositeOperation = "destination-over";

    //set background color
    context.fillStyle = "#FFFFFF";

    // draw background/rectangle on entire canvas - substitue code values 
    context.fillRect(0, 0, width, height);  
    var tempCanvas = document.createElement("canvas"),
        tCtx = tempCanvas.getContext("2d");
    
    tempCanvas.width = 640;
    tempCanvas.height = 480;
    
    tCtx.drawImage(canvas[0],0,0);
    
    // write on screen
    var img = tempCanvas.toDataURL("image/png");
    document.write('<a href="'+img+'"><img src="'+img+'"/></a>');
  })

  var image = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");  
  window.location.href=image; // save locally

  
  model = await facemesh.load({maxFaces: state.maxFaces});
  renderPrediction();

  if (renderPointcloud) {
    document.querySelector('#scatter-gl-container').style =
        `width: ${VIDEO_SIZE}px; height: ${VIDEO_SIZE}px;`;

    scatterGL = new ScatterGL(
        document.querySelector('#scatter-gl-container'),
        {'rotateOnStart': false, 'selectEnabled': false});
  }
};

main();

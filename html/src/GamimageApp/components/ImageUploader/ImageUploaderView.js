import { ImageUploaderModel } from './ImageUploaderModel.js';

import * as d3 from 'd3';

export function ImageUploaderView(rootObj, app) {
    let self = this;
    self.app = app;
    self.state = app.model.state;
    self.rootObj = rootObj;
    self.objs = {}


    self.setupInitialView = function () {

        
        // if (self.state.loadedCode !== undefined) {
        //     self.rootObj.select('#imageDropArea')
        //         .selectAll('*')
        //         .remove();
        // } else {
        // }



    }

    self.setupDragArea = function (state) {

        // ************************ Drag and drop ***************** //
        let dropArea = document.getElementById("imageDropArea")

            // Prevent default drag behaviors
            ;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false)
                document.body.addEventListener(eventName, preventDefaults, false)
            })

            // Highlight drop area when item is dragged over it
            ;['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false)
            })

            ;['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false)
            })

        // Handle dropped files
        dropArea.addEventListener('drop', handleDrop, false)

        function preventDefaults(e) {
            e.preventDefault()
            e.stopPropagation()
        }

        function highlight(e) {
            dropArea.classList.add('highlight')
        }

        function unhighlight(e) {
            dropArea.classList.remove('active')
        }

        function handleDrop(e) {
            var dt = e.dataTransfer
            var files = dt.files

            handleFiles(files)
        }

        let uploadProgress = []
        // let progressBar = document.getElementById('progress-bar')

        function initializeProgress(numFiles) {
            // progressBar.value = 0
            uploadProgress = []

            for (let i = numFiles; i > 0; i--) {
                uploadProgress.push(0)
            }
        }

        function updateProgress(fileNumber, percent) {
            uploadProgress[fileNumber] = percent
            let total = uploadProgress.reduce((tot, curr) => tot + curr, 0) / uploadProgress.length
            console.debug('update', fileNumber, percent, total)
            // progressBar.value = total
        }

        function handleFiles(files) {
            files = [...files]
            initializeProgress(files.length)
            files.forEach(uploadFile)
            files.forEach(previewFile)
        }

        d3.select('#fileElem')
            .on('change', function(){
                handleFiles(this.files);
            })

        function previewFile(file) {
            let reader = new FileReader()
            reader.readAsDataURL(file)
            reader.onloadend = function () {
                d3.selectAll('img').remove();
                let img = document.createElement('img')
                img.src = reader.result
                document.getElementById('gallery').appendChild(img)
            }
        }

        function uploadFile(file, i) {
            var url = self.app.model.state.apiBaseURL + 'uploadImage'
            var xhr = new XMLHttpRequest()
            var formData = new FormData()
            xhr.open('POST', url, true)
            xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest')

            // Update progress (can be used to show progress indicator)
            xhr.upload.addEventListener("progress", function (e) {
                updateProgress(i, (e.loaded * 100.0 / e.total) || 100)
            })

            xhr.addEventListener('readystatechange', function (e) {
                if (xhr.readyState == 4 && xhr.status == 200) {
                    updateProgress(i, 100);
                    self.app.model.takeAction('image has been uploaded', JSON.parse(xhr.response));
                }
                else if (xhr.readyState == 4 && xhr.status != 200) {
                    // Error. Inform the user
                }
            })

            formData.append('image_type', 'uploadedImage')
            formData.append('code', self.app.model.state.codeMeta.code)
            formData.append('upload_preset', 'ujpu6gyk')
            formData.append('file', file)
            xhr.send(formData)
        }

    }

    self.draw = function (state, settings, data) {
        if (self.state.loadedCode) {

            let inputContainer = self.rootObj.select('.inputContainer.upload');
            inputContainer.select('#imageDropArea')
                .remove();
            
            if (inputContainer.select('.uploadedImage').empty()) {
                inputContainer
                    .append('img')
                    .classed('uploadedImage', true)
                    .attr('src', d => './userContent/'
                        + state.codeMeta.code
                        + '/input/uploadedImage.png' + '?'
                        + new Date().getTime())
            }

        } else {
            self.setupDragArea(self.state)
        }
    }

    self.setupInitialView();
}
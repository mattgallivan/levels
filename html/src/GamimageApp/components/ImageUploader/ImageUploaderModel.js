// https://www.smashingmagazine.com/2018/01/drag-drop-file-uploader-vanilla-js/
// https://codepen.io/joezimjs/pen/yPWQbd

export function ImageUploaderModel(app) {
    let self = this;
    self.app = app;
    self.state = app.model.state;
    self.settings = app.model.settings;
    self.data = app.model.data;

    self.updateModel = function () {

    }

}
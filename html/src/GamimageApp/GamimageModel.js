import { GamimageApp } from './GamimageApp.js';

import { ImageUploaderModel } from './components/ImageUploader/ImageUploaderModel.js'
import { RoutingModel } from './components/Routing/RoutingModel.js'
import { GeneratedOutputModel } from './components/GeneratedOutput/GeneratedOutputModel.js'

export function GamimageModel(app, initialState, initialSettings, initialData) {
    let self = this;
    self.app = app;
    self.state = initialState;
    self.settings = initialSettings;
    self.data = initialData;


    self.takeAction = function (action, val) {

        if (action === 'image has been uploaded') {
            // Code not provided, get a new code
            fetch(self.state.apiBaseURL + 'turnImageIntoGameContent', {
                    method: 'POST', // *GET, POST, PUT, DELETE, etc.
                    mode: 'cors', // no-cors, *cors, same-origin
                    cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
                    credentials: 'same-origin', // include, *same-origin, omit
                    headers: {
                        'Content-Type': 'application/json'
                        // 'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: JSON.stringify(val)
                })
                .then(response => response.json())
                .then(config => {
                    self.state.generated = {
                        config
                    }
                    app.view.draw(self.state, self.settings);
                })
        }

    }

    self.resetPluginState = function () {

        // Run routing model first.
        self.routingModel.fetchRouteFirst(() => {
            self.updateModelsAfterRoutingUpdated();
            app.view.draw(self.state, self.settings, self.data);
        });
        
    }

    self.updateModelsAfterRoutingUpdated = function() {
        // Run component models
        self.modelComponents.forEach(componentMeta => {
            componentMeta.component.updateModel(self.state, self.settings);
        })        
    }

    self.startPlugin = function (view) {
        self.view = view;

        self.routingModel = new RoutingModel(app);

        self.modelComponents = [
            { key: 'routing', component: self.routingModel },
            { key: 'imageUploader', component: new ImageUploaderModel(app) },
            { key: 'generatedOutput', component: new GeneratedOutputModel(app) },
        ];        

        self.resetPluginState();
    }

}
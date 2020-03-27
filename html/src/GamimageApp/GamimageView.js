import { GamimageApp } from './GamimageApp.js';

import { ImageUploaderView } from './components/ImageUploader/ImageUploaderView.js'
import { GeneratedOutputView } from './components/GeneratedOutput/GeneratedOutputView.js'
import { RoutingView } from './components/Routing/RoutingView.js'

import * as d3 from 'd3';

export function GamimageView(app, rootSel, settings) {
    let self = this;
    self.app = app;
    self.rootSel = rootSel;
    self.rootObj = d3.select(rootSel);
    self.objs = {}



    self.setupInitialView = function () {
        // let svgObj = d3.select(self.root).append('svg');
        // self.objs.svg = svgObj;
        // svgObj.attr('width', settings.view.svg.width + 'px');
        // svgObj.attr('height', settings.view.svg.height + 'px');

        let inputMethods = [
            {
                id: "upload",
                tabTitle: "Upload Image"
            },
            {
                id: "drawing",
                tabTitle: "Draw!"
            }
        ];


        let appContainer = self.rootObj;

        // let tabButtonContainer = appContainer.select('.tabButtonsContainer').selectAll('.tabButtonContainer')
        //     .data(inputMethods)
        //     .enter()
        //     .append('div')
        //     .classed('tabButtonContainer', true)

        // tabButtonContainer
        //     .append('div')
        //     .classed('tabTitle', true)
        //     .text(d => d.tabTitle)


        self.viewComponents = [
            { key: 'routing', component: new RoutingView(self.rootObj, app) },
            { key: 'imageUploader', component: new ImageUploaderView(self.rootObj, app) }, ,
            { key: 'generatedOutput', component: new GeneratedOutputView(self.rootObj, app) },
        ];

    }

    self.draw = function (state, settings, data) {


        // Run component views
        self.viewComponents.forEach(componentMeta => {
            componentMeta.component.draw(state, settings, data);
        })


    }

    self.setupInitialView();
}
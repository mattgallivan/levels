import { RoutingModel } from './RoutingModel.js';

import * as d3 from 'd3';

export function RoutingView(rootObj, app) {
    let self = this;
    self.app = app;
    self.rootObj = rootObj;
    self.objs = {}
 

    self.setupInitialView = function () {


    }

    self.draw = function (state, settings, data) {
        let rootObj = self.rootObj;
        rootObj.select('.currCodeURL')
            .text(window.location.href + '?' + state.codeMeta.code);
    }

    self.setupInitialView();
}
import { RoutingModel } from './RoutingModel.js';
import QRCode from 'qrcode'

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

        let pageUrl = window.location.href + '?' + state.codeMeta.code;
        rootObj.select('.currCodeURL')
            .text(pageUrl);

        QRCode.toCanvas(rootObj.select('.qrCanvas').node(), pageUrl , function (error) {
            if (error) console.error(error)
            console.log('success!');
        })
    }

    self.setupInitialView();
}
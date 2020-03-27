import { GamimageView } from './GamimageView.js';
import { GamimageModel } from './GamimageModel.js';
import '../../assets/css/imageUploader.scss';
import '../../assets/css/main.scss';

export function GamimageApp(rootSel) {
    let self = this;

    self.rootSel = rootSel;

    let initialState = {

    }

    let initialSettings = {
        api: {
            port: 5000,
            domain: document.domain,
            protocol: location.protocol
        }
    }

    self.model = new GamimageModel(this, initialState, initialSettings, '');
    self.view = new GamimageView(this, rootSel, initialSettings);

    self.model.startPlugin(self.view);   
}


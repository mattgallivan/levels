import { GeneratedOutputModel } from './GeneratedOutputModel.js';

import * as d3 from 'd3';

export function GeneratedOutputView(rootObj, app) {
    let self = this;
    self.app = app;
    self.rootObj = rootObj;
    self.objs = {}


    self.setupInitialView = function () {

    }

    self.draw = function (state, settings, data) {

        if (state.generated && state.generated.games) {
            
            let outputGameSections = self.rootObj
                .select('.generatedOutputContainer')
                .selectAll('.gameSection')
                .data(state.generated.games);
    
            let outputGameSectionsEntered = 
                outputGameSections.enter()
                .append('div')
                .classed('gameSection', true);
                
            outputGameSections = outputGameSectionsEntered.merge(outputGameSections);
    
            outputGameSections.each(function(gameName, i) {
                let gameSectionContainer = d3.select(this);
                gameSectionContainer.selectAll('*').remove();
    
                gameSectionContainer.append('div').text(gameName);

                gameSectionContainer
                    .selectAll('.outputLevelImage')
                    .data([
                        'histogram_match_16px_uploadedImage',
                        'histogram_match_32px_uploadedImage',
                        'histogram_match_50px_uploadedImage',
                        'img_match_16px_uploadedImage',
                        'img_match_32px_uploadedImage',
                        'img_match_50px_uploadedImage',
                    ])
                    .enter()
                    .append('img')
                    .attr('src', d => './userContent/' 
                    + state.codeMeta.code 
                    + '/output/uploadedImage.png/games/' 
                    + gameName + '/' + d + '.png' + '?' 
                    + new Date().getTime())

            })

        }
    }

    self.setupInitialView();
}
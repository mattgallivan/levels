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

        if (state.generated && state.generated.config) {
            
            let outputGameSections = self.rootObj
                .select('.generatedOutputContainer')
                .selectAll('.gameSection')
                .data(state.generated.config.gamesData);
    
            let outputGameSectionsEntered = 
                outputGameSections.enter()
                .append('div')
                .classed('gameSection', true)
                .classed('gameSection-not-0', (d,i) => i !== 0);
                
            outputGameSections = outputGameSectionsEntered.merge(outputGameSections);
    
            outputGameSections.each(function(gameDataMeta, i) {
                let gameSectionContainer = d3.select(this);
                gameSectionContainer.selectAll('*').remove();
    
                gameSectionContainer.append('div').text(gameDataMeta.game_info.shortname);


                let outputList = [];

                // Tile-based
                let tileSection = gameSectionContainer
                    .append('div')
                    .classed('tileSection', true);

                tileSection
                    .append('div')
                    .classed('tileSectionTitle', true)
                    .text('tile-based matching');

                for (let matchMethod of Object.keys(gameDataMeta.conversions['tile-based matching'])) {

                    let matchMethodSection = tileSection
                        .append('div')
                        .classed('matchMethodSection', true);

                    matchMethodSection
                        .append('div')
                        .classed('matchMethodSectionTitle', true)
                        .text(matchMethod)
                    
                    let tileSizeList = [];

                    for (let tileSize of gameDataMeta.conversions['tile-based matching'][matchMethod]) {
                        let outputData = {
                            filename: matchMethod + '_match_' + tileSize + 'px_uploadedImage',
                            matchMethod, tileSize,
                            conversionType: 'tile-based matching'
                        };
                        outputList.push(outputData)
                        tileSizeList.push(outputData)
                    }


                    let outputMiniDiv = matchMethodSection.selectAll('.outputMiniDiv')
                        .data(tileSizeList)
                        .enter()
                        .append('div')
                        .classed('outputMiniDiv', true)
                        .on('click', (d,i) => {
                            
                            let selectedContainer = self.rootObj
                                .select('.selectedOutputContainer');

                            selectedContainer.selectAll('*').remove();

                            selectedContainer
                                .append('img')
                                .classed('selectedOutputMainImg', true)
                                .attr('src', './userContent/'
                                    + state.codeMeta.code
                                    + '/output/uploadedImage.png/games/'
                                    + gameDataMeta.game_info['path-friendly-name']
                                    + '/'
                                    + d.filename + '.png' + '?'
                                    + new Date().getTime())
                                .on('click', () => {
                                    window.open('./userContent/'
                                        + state.codeMeta.code
                                        + '/output/uploadedImage.png/games/'
                                        + gameDataMeta.game_info['path-friendly-name']
                                        + '/'
                                        + d.filename + '.png' + '?'
                                        + new Date().getTime()
                                        , '_blank');
                                })

                            selectedContainer
                                .append('div')
                                .classed('asciiLevel', true)
                                .text('ASCII Level')
                                .on('click', () => {
                                    window.open('./userContent/'
                                        + state.codeMeta.code
                                        + '/output/uploadedImage.png/games/'
                                        + gameDataMeta.game_info['path-friendly-name']
                                        + '/'
                                        + d.filename + '.txt' + '?'
                                        + new Date().getTime()
                                        , '_blank');
                                })
                        })
                        ;

                    outputMiniDiv
                        .append('img')
                        .classed('outputLevelImage', true)
                        .attr('src', d => './userContent/'
                            + state.codeMeta.code
                            + '/output/uploadedImage.png/games/'
                            + gameDataMeta.game_info['path-friendly-name']
                            + '/'
                            + d.filename + '.png' + '?'
                            + new Date().getTime())

                    outputMiniDiv
                        .append('div')
                        .classed('tileSizeText', true)
                        .text(d=>d.tileSize+'px');
                }
                


                // gameSectionContainer
                //     .selectAll('.outputLevelImage')
                //     .data(outputList)
                //     .enter()
                //     .append('img')
                //     .classed('outputLevelImage', true)
                //     .attr('src', d => './userContent/' 
                //     + state.codeMeta.code 
                //     + '/output/uploadedImage.png/games/' 
                //         + gameDataMeta.game_info['path-friendly-name'] 
                //         + '/' 
                //         + d.filename + '.png' + '?' 
                //     + new Date().getTime())

            })

        }
    }

    self.setupInitialView();
}
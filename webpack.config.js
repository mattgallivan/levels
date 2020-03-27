const CopyWebpackPlugin = require('copy-webpack-plugin');
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const HtmlWebPackPlugin = require("html-webpack-plugin");

module.exports = {

    entry: './html/src/main.js',
    devtool: 'cheap-module-source-map',

    module: {
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: {
                    loader: "babel-loader"
                }
            },
            {
                test: /\.html$/,
                use: [
                    {
                        loader: "html-loader",
                        options: { minimize: true }
                    }
                ]
            },
            {
                test: /\.scss$/,
                use: [
                    "style-loader",
                    MiniCssExtractPlugin.loader,
                    "css-loader",
                    "sass-loader"
                ]
            },
            {
                test: /\.(svg|png|jpg|gif)$/,
                use: {
                    loader: "file-loader",
                    options: {
                        esModule: false
                    }
                }
            }
        ]
    },
    plugins: [
        new MiniCssExtractPlugin({
            filename: 'main.css',
        }),
        new HtmlWebPackPlugin({
            template: "./html/index.html",
            filename: "./index.html"
        }),
        new CopyWebpackPlugin([
            { from: 'data', to: 'data' },
        ]),
    ]
};
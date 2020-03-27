export function RoutingModel(app) {
    let self = this;
    self.app = app;
    self.state = app.model.state;
    self.settings = app.model.settings;
    self.data = app.model.data;

    self.updateModel = function () {


        // console.log();
        // debugger;

    }

    self.fetchRouteFirst = function(cbf) {

        let settings = self.settings;
        let apiBaseURL = settings.api.protocol
            + '//' + settings.api.domain
            + ':' + settings.api.port + '/api/';

        self.state.apiBaseURL = apiBaseURL;

        // Check if Code is provided
        let codeProvided = window.location.search !== "";
        let loadCode = codeProvided && window.location.search.substring(1)[0] !== '!'

        if (loadCode) {
            let code = window.location.search.substring(1);
            let codeMeta = {code};
            fetch(apiBaseURL + 'obtainConfigForCode', {
                method: 'POST', // *GET, POST, PUT, DELETE, etc.
                mode: 'cors', // no-cors, *cors, same-origin
                cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
                credentials: 'same-origin', // include, *same-origin, omit
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(codeMeta)
            })
                .then(response => response.json())
                .then(config => {
                    self.updateCodeMeta(codeMeta, config)
                    self.state.loadedCode = true;
                    window.history.replaceState({}, document.title, "/");
                    cbf()
                })
            
            // cbf()
        } else {
            // Get a new code or ask for a specific one
            let newCode = '';
            if (codeProvided) {
                newCode = window.location.search.substring(1).replace(/!/g, '');
            }
            fetch(apiBaseURL + 'generateNewCode', {
                method: 'POST', // *GET, POST, PUT, DELETE, etc.
                mode: 'cors', // no-cors, *cors, same-origin
                cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
                credentials: 'same-origin', // include, *same-origin, omit
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ code: newCode})
            })
                .then(response => response.json())
                .then(codeMeta => {
                    self.updateCodeMeta(codeMeta);
                    self.state.loadedCode = false;
                    window.history.replaceState({}, document.title, "/");
                    cbf()
                })
        }

    }

    self.updateCodeMeta = function (codeMeta, config) {
        self.state['codeMeta'] = codeMeta;

        if (config !== undefined) {
            self.state.generated = {
                config
            }
        }

    }

}
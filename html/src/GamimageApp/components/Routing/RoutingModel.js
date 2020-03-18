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

        if (codeProvided) {
            let id = window.location.search.substring(1);
            window.history.replaceState({}, document.title, "/");
            cbf()
        } else {
            // Code not provided, get a new code
            fetch(apiBaseURL + 'generateNewCode')
                .then(response => response.json())
                .then(codeMeta => {
                    self.updateCodeMeta(codeMeta)
                    cbf()
                })
        }

    }

    self.updateCodeMeta = function (codeMeta) {
        self.state['codeMeta'] = codeMeta;
    }

}
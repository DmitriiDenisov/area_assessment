// ============================= POLYFILLS ===============================
(function () {
    // MATCHES polyfill
    if (!Element.prototype.matches) {
        Element.prototype.matches = 
            Element.prototype.matchesSelector || 
            Element.prototype.mozMatchesSelector ||
            Element.prototype.msMatchesSelector || 
            Element.prototype.oMatchesSelector || 
            Element.prototype.webkitMatchesSelector ||
            function(s) {
                var matches = (this.document || this.ownerDocument).querySelectorAll(s),
                    i = matches.length;
                while (--i >= 0 && matches.item(i) !== this) {}
                return i > -1;
            };
    }

    // CLOSEST polyfill
    if (window.Element && !Element.prototype.closest) {
        Element.prototype.closest = 
        function(s) {
            var matches = (this.document || this.ownerDocument).querySelectorAll(s),
                i,
                el = this;
            do {
                i = matches.length;
                while (--i >= 0 && matches.item(i) !== el) {};
            } while ((i < 0) && (el = el.parentElement)); 
            return el;
        };
    }
})();

(function () {
    // =================================================
    // HELPERS
    // =================================================
    function clone(obj) {
        var copy;

        // Handle the 3 simple types, and null or undefined
        if (null == obj || "object" != typeof obj) return obj;

        // Handle Date
        if (obj instanceof Date) {
            copy = new Date();
            copy.setTime(obj.getTime());
            return copy;
        }

        // Handle Array
        if (obj instanceof Array) {
            copy = [];
            for (var i = 0, len = obj.length; i < len; i++) {
                copy[i] = clone(obj[i]);
            }
            return copy;
        }

        // Handle Object
        if (obj instanceof Object) {
            copy = {};
            for (var attr in obj) {
                if (obj.hasOwnProperty(attr)) copy[attr] = clone(obj[attr]);
            }
            return copy;
        }

        throw new Error("Unable to copy obj! Its type isn't supported.");
    }
    // ==========================================================
    // ==========================================================

    // mapboxgl.accessToken = 'pk.eyJ1Ijoib21hbmd1dG92IiwiYSI6ImNpenpvNWtmajAwMjgzMnBmYXp6enhuOTIifQ.KwVxe-y3-a_BeY-uHqBoag';
    // new pk.eyJ1IjoidHVybmlrayIsImEiOiJjanlmcmJpazYwMDc1M2dsZ21tbWhqNnU5In0._LsflxOVYYFfx4CRZ35Wfw
    // nick // 'pk.eyJ1IjoidHVybmlrayIsImEiOiJjanlmcmJpazYwMDc1M2dsZ21tbWhqNnU5In0._LsflxOVYYFfx4CRZ35Wfw';
    mapboxgl.accessToken = 'pk.eyJ1IjoiZG1pdHJpaWRlbmlzb3YiLCJhIjoiY2p5eGMwam15MTV1MjNucGhyNTdicnNyMSJ9.MTiFOwhPS4ms_cYTwzmRTg';

    var mapRoot = document.getElementById('map-root');
    var mapWrapper = mapRoot.querySelector('.map__wrapper');
    var canvas;

    var mapSettings = {
        cursor: '',
        style: {
            baseURI: 'mapbox://styles/dmitriidenisov/', // mapbox://styles/turnikk/
            defaultStyle: 'cjzgymzpz0oqv1coi6qd5vkk5' //'cjzbkqgjy0a571cqcyaxoaxbn' // cjyfshbbl1hi01crs5gu9z6pw
        }
    };

    var map = new mapboxgl.Map({
        container: 'map',
        style: mapSettings.style.baseURI + mapSettings.style.defaultStyle,
        attributionControl: false
    });

    map.addControl(new mapboxgl.ScaleControl(), 'bottom-left');
    map.addControl(new mapboxgl.NavigationControl(), 'top-right');

    var layers = {
        'mecca-mapbox-newmodel-rect-7jr76k': {
            id: 'mecca-mapbox-newmodel-rect-7jr76k',
            name: 'Mecca Mapbox New Model Rectangle',
            visible: true,
            color: '#2f0aff'
        },
        'mecca-mapbox-newmodel-default-29tzxv': {
            id: 'mecca-mapbox-newmodel-default-29tzxv',
            name: 'Mecca Mapbox New Model Default',
            visible: true,
            color: '#8aff14'
        },
        'mecca_mapbox jaccard model': {
            id: 'mecca_mapbox jaccard model',
            visible: true,
            name: 'Mecca Mapbox Jaccard model',
            color: '#fd1795'
        },
        'mapbox sakaka jaccard model': {
            id: 'mapbox sakaka jaccard model',
            visible: true,
            name: 'Mapbox Sakaka Jaccard Model',
            color: '#f57e00'
        }
        // 'bing_maps jaccard model': {
        //     id: 'bing_maps jaccard model',
        //     visible: true,
        //     name: 'Bing_maps Jaccard model',
        //     color: '#e20808'
        // },
        // 'initial_image_from_server jaccard model': {
        //     id: 'initial_image_from_server jaccard model',
        //     visible: true,
        //     name: 'Initial image from server Jaccard model',
        //     color: '#8b0985'
        // }
    };
    var layersIds = Object.keys(layers).map(function (layerId) {
        return layerId;
    });

    // ===========================================================================
    // ===========================================================================
    // OVERLAY MODULE
    // ===========================================================================
    // ===========================================================================
    var overlayModule = (function () {
        function show() {
            mapRoot.classList.remove('loaded');
            mapRoot.classList.add('loading');
        }

        function hide() {
            mapRoot.classList.remove('loading');
            mapRoot.classList.add('loaded');
        }

        return {
            show: show,
            hide: hide
        };
    })();
    // ===========================================================================
    // end OVERLAY MODULE
    // ===========================================================================

    // ===========================================================================
    // ===========================================================================
    // FEATURES MODULE
    // ===========================================================================
    // ===========================================================================
    var featuresModule = (function () {
        var props = [
            { name: 'name', type: 'string' },
            { name: 'area', type: 'number', append: ' m<sup>2</sup>' },
        ];

        function renderPropItem(prop, feature) {
            var val = feature.properties[prop.name];

            if (prop.type === 'number') {
                val = (val != null) ? val.toFixed(2) : 'N/A';
            }

            if (prop.append) {
                val += prop.append;
            }

            return '' +
                '<div class="feature__item">' + 
                    '<dt class="feature__prop">' + prop.name + '</dt>' + 
                    '<dd class="feature__val">' + val + '</dd>' +
                '</div>'
        }

        function renderAllProps(feature) {
            var tmpl = '';

            props.forEach(function (prop) {
                tmpl += renderPropItem(prop, feature);
            });

            return tmpl;
        }

        function openPopup(point, feature) {
            var popup = new mapboxgl.Popup();

            popup.setLngLat(map.unproject(point))
                .setHTML(
                    '<div className="feature">' +
                        '<h3 class="feature__title">Building info</h3>' +
                        '<dl className="feature__list">' + 
                            renderAllProps(feature) +
                        '</dl>' +
                    '</div>')
                .addTo(map);
        }

        function saveInfo(callback) {
            var initialZoom = map.getZoom();
            var center = map.getCenter();

            map.once('zoomend', function() {
                var interval = setInterval(function () {
                    if (!map.loaded()) {
                        return;
                    }
                    
                    layersIds.forEach(function (id) {
                        var layer = layers[id];
                        var feature = map.queryRenderedFeatures({ layers: [id] })[0];

                        if (!feature) {
                            return;
                        }

                        layer.featuresQuantity = feature.properties.totalCount;
                        layer.featuresArea = feature.properties.totalArea;
                    });

                    map.setZoom(initialZoom);
                    map.setCenter(center);

                    callback();
                    clearInterval(interval);
                }, 10);
            });

            map.setZoom(11);
        }

        // module exports
        return {
            openPopup: openPopup,
            saveInfo: saveInfo
        };
    })();
    // ===========================================================================
    // end FEATURES MODULE
    // ===========================================================================


    // ===========================================================================
    // ===========================================================================
    // LAYERS MODULE
    // ===========================================================================
    // ===========================================================================
    var layersModule = (function () {
        var mapInfo = document.getElementById('map-info');
        var opacityProperties = ['fill-opacity'];

        function renderStats(data) {
            function renderItems() {
                var itemsTmpl = '';

                data.forEach(function (item) {
                    var value = Math.round(item.value);

                    if (isNaN(value)) {
                        value = 'No data';
                    }

                    itemsTmpl += 
                        '<div class="layer__item">' +
                            '<dt class="layer__name">' + item.name + '</dt>' +
                            '<dd class="layer__val">' + value + '</dd>' +
                        '</div>';
                });

                return itemsTmpl;
            }

            var tmpl = '' +
                '<dl class="layer__stats">' +
                    renderItems() +
                '</dl>';
            
            return tmpl;
        }

        function renderControls(layer) {
            var heatmap = !!layer.heatmap;

            return '' +
                '<div class="layer__controls">' +
                    '<div class="layer__control">' +
                        '<div class="layer__label">Visibility</div>' +
                        '<label class="switcher">' +
                            '<input type="checkbox" class="switcher__input layer__visibility" checked>' +
                            '<span class="switcher__icon"></span>' +
                        '</label>' +
                    '</div>' +
                    '<div class="layer__control">' +
                        '<div class="layer__label">' +
                            'Opacity' +
                        '</div>' +
                        '<input type="range" value="100" class="range-slider layer__opacity">' +
                    '</div>' +
                    // (heatmap ?
                    // '<div class="layer__control layer__control_full-width">' +
                    //     '<div class="layer__label">Urbanisation Index</div>' +
                    //     '<label class="switcher">' +
                    //         '<input type="checkbox" class="switcher__input layer__heatmap">' +
                    //         '<span class="switcher__icon"></span>' +
                    //     '</label>' +
                    // '</div>' :
                    // '') +
                '</div>';
        }

        function removeSelectedArea(wrapper) {
            var selectedArea = wrapper.querySelector('.layer_selected');

            if (selectedArea) {
                selectedArea.parentElement.removeChild(selectedArea);
            }
        }

        function renderSelectedArea(features, wrapper) {
            var values = {
                total: 0
            };
            var tmpl = '';
            var dataForRender = [];

            removeSelectedArea(wrapper);

            if (!features || !features.length) {
                return;
            }
            
            features.forEach(function (feature) {
                var layer = feature.layer.id;

                if (!values[layer]) {
                    values[layer] = 0;
                }

                values[layer] += feature.properties.area;
                values.total += feature.properties.area;
            });

            layersIds.forEach(function (id) {
                var layer = layers[id];

                dataForRender.push({
                    name: layer.name + ' (m<sup>2</sup>)',
                    value: values[layer.id] ? values[layer.id] : 0
                });
            });

            dataForRender.push({
                name: 'Total (m<sup>2</sup>)',
                value: values.total
            });

            tmpl +=
                '<section class="layer layer_selected">' +
                    '<h3 class="layer__title">Area under selection</h3>' +
                    renderStats(dataForRender) +
                '</section>';

            wrapper.insertAdjacentHTML('beforeend', tmpl);
            map.resize();
        }

        function renderLayersInfo() {
            var tmpl = '';

            layersIds.forEach(function (id) {
                var layer = layers[id];
                var stats = [
                    // { name: 'Quantity', value: layer.featuresQuantity },
                    // { name: 'Area (m<sup>2</sup>)', value: layer.featuresArea }
                ];

                tmpl +=
                    '<section class="layer" data-id="' + id + '">' +
                    // '<div class="layer-name-color">' +
                    '<span class="color-icon" style="background: ' +
                    layer.color +
                    '"></span>' +
                    '<h3 class="layer__title">' + layer.name + '</h3>' +
                    // '</div>'+
                    renderControls(layer) +
                    '</section>'
            });

            mapInfo.insertAdjacentHTML('beforeend', tmpl);
            map.resize();
        }

        function getLayerEl(layerId) {
            return mapInfo.querySelector('.layer[data-id="' + layerId + '"]');
        }

        function setupOpacityFilter(layerId) {
            var layer = getLayerEl(layerId);
            var opacityFilter;

            if (!layer) {
                return;
            }

            opacityFilter = layer.querySelector('.layer__opacity');

            opacityProperties.forEach(function (property) {
                var layerOpacity = map.getPaintProperty(layerId, property);
                console.log(layerId, layerOpacity);
                if (!layerOpacity && layerOpacity !== 0) {
                    opacityFilter.disabled = true;
                    return;
                }

                opacityFilter.disabled = false;
                opacityFilter.value = Math.round(layerOpacity * 100);
            });
        }

        function setOpacity(layerId, val) {
            opacityProperties.forEach(function (property) {
                var propertyVal = map.getPaintProperty(layerId, property);
                if (!propertyVal && propertyVal !== 0) {
                    return;
                }

                map.setPaintProperty(layerId, property, val);
                console.log(map.getPaintProperty(layerId, property));
            });
        }

        function setVisibility(visibility, layerId) {
            var value = visibility ? 'visible' : 'none';
            layers[layerId].visible = visibility;
            map.setLayoutProperty(layerId, 'visibility', value);
        }

        document.addEventListener('change', function (e) {
            var target = e.target;
            var layer = target.closest('.layer');

            if (!layer) {
                return;
            }

            var layerId = layer.getAttribute('data-id');
            var visibilitySwitcher = target.closest('.layer__visibility');
            var opacitySlider = target.closest('.layer__opacity');
            var heatmapSwitcher = target.closest('.layer__heatmap');

            // Show or hide layer when checkbox toggle
            if (visibilitySwitcher) {
                setVisibility(visibilitySwitcher.checked, layerId);
            }

            // Change layer opacity when drag slider
            if (opacitySlider) {
                var opacity = parseInt(opacitySlider.value, 10) / 100;
                setOpacity(layerId, opacity);
            }

            if (heatmapSwitcher) {
                // heatmapModule.setVisibility(layerId, heatmapSwitcher.checked);
            }
        });

        map.on('custom.changestyle', function () {
            layersIds.forEach(function (id) {
                setVisibility(layers[id].visible, id);
                setupOpacityFilter(id);
            });
        });

        function init() {
            renderLayersInfo();
            layersIds.forEach(setupOpacityFilter);
        }

        // module exports
        return {
            init: init,
            renderSelectedArea: renderSelectedArea,
            removeSelectedArea: removeSelectedArea,
            setVisibility: setVisibility
        };
    })();

    // ===========================================================================
    // ===========================================================================
    // MOVES MODULE
    // ===========================================================================
    // ===========================================================================
    var movesModule = (function () {
        var movesInputName = 'map-moves';
        var movesWrapper = document.getElementById(movesInputName);
        var moves = {
            Mecca: {
                lng: 39.826694,
                lat: 21.422751
            },
            Dawmat: {
                lng: 39.8726,
                lat: 29.80138
            }
        }
        
        function render() {
            var movesTmpl ='';

            for (var city in moves) {
                if (!moves.hasOwnProperty(city)) {
                    continue;
                }
                var coords = moves[city];

                movesTmpl +=
                    '<button class="mapboxgl-ctrl-icon move-button" ' +
                    'type="button" ' +
                    'id="' +
                    city +
                    '">' +
                     city +
                    '</button>';
            }
            movesWrapper.insertAdjacentHTML('beforeend',movesTmpl);
        }

        function init() {
            render();
        }

        function moveMap(coords) {
            map.setCenter(coords);
        }

        document.addEventListener('click', function (e) {
            var target = e.target;
            var moveBtnPressed = target.closest('.move-button');

            if (!moveBtnPressed) {
                return;
            }
            moveMap(moves[moveBtnPressed['id']]);
        });
        return {
            init: init,
            move: moveMap,
            moves: moves,
        };
    })();
    // ===========================================================================
    // ===========================================================================
    // STYLES MODULE
    // ===========================================================================
    // ===========================================================================
    var stylesModule = (function () {
        var stylesWrapper = document.getElementById('map-styles');
        var INPUT_NAME = 'map-styles';
        var styles = {
            satellite: {
                url: 'cj0tpd64o00j82rnphr76axim',
                name: 'Satellite'
            },
        };

        // function render() {
        //     var tmpl = '';
        //
        //     for (var prop in styles) {
        //         if (!styles.hasOwnProperty(prop)) {
        //             continue;
        //         }
        //
        //         var style = styles[prop];
        //
        //         tmpl +=
        //             '<label class="radio-btn">' +
        //                 '<input ' +
        //                     'name="' + INPUT_NAME + '" ' +
        //                     'type="radio" ' +
        //                     'class="radio-btn__input" ' +
        //                     'value="' + prop + '" ' +
        //                     (style.url === mapSettings.style.defaultStyle ? 'checked' : '') +
        //                 '>' +
        //                 '<span class="radio-btn__text">' + style.name + '</span>' +
        //             '</label> ';
        //     }
        //
        //     stylesWrapper.insertAdjacentHTML('beforeend', tmpl);
        // }

        function syncRadioButtons(checkedVal) {
            var radioButtons = stylesWrapper.querySelectorAll('.radio-btn__input');

            Array.prototype.forEach.call(radioButtons, function (btn) {
                if (checkedVal === btn.value) {
                    if (!btn.checked) {
                        btn.checked = true;
                    }
                } else {
                    if (btn.checked) {
                        btn.checked = false;
                    }
                }
            });
        }

        function setStyle(style) {
            overlayModule.show();
            map.once('style.load', function () {
                overlayModule.hide();

                map.fire('custom.changestyle', {
                    style: style
                });
            });

            map.setStyle(mapSettings.style.baseURI + styles[style.toLowerCase()].url);
            syncRadioButtons(style);
        }

        document.addEventListener('change', function (e) {
            var target = e.target;
            var styleChanged = target.closest('.radio-btn__input[name="' + INPUT_NAME + '"]');

            if (!styleChanged) {
                return;
            }

            setStyle(styleChanged.value);
        });

        function init() {
            // render();
        }

        return {
            set: setStyle,
            init: init
        };
    })();
    // ===========================================================================
    // end STYLES MODULE
    // ===========================================================================


    // ===========================================================================
    // ===========================================================================
    // MODES MODULE
    // ===========================================================================
    // ===========================================================================
    var modesModule = (function () {
        var modes = {
            // ruler: {
            //     module: rulerModule,
            //     title: 'Ruler mode'
            // },
            // selected: {
            //     module: selectedAreaModule,
            //     title: 'Selection mode'
            // }
        };
        var activeMode = null;

        function forEachMode(callback) {
            for (var modeName in modes) {
                if (!modes.hasOwnProperty(modeName)) {
                    continue;
                }

                callback(modeName);
            }
        }

        function disableAllModes() {
            modes[activeMode].module.disable();
            activeMode = null;
            // tipsModule.clear();
        }

        function renderControls() {
            var tmpl = '';
            var controlsWrapper = document.getElementById('map-modes');

            forEachMode(function (mode) {
                tmpl +=
                    '<button class="control-btn" data-mode="' + mode + '">' +
                        modes[mode].title +
                    '</button>';
            });

            controlsWrapper.insertAdjacentHTML('beforeend', tmpl);
        }

        function setMode(mode) {
            if (!mode in modes) {
                return;
            }

            if (activeMode) {
                modes[activeMode].module.disable();
            }

            // tipsModule.clear();
            activeMode = mode;
            modes[mode].module.enable();
        }

        // renderControls();

        document.addEventListener('click', function (e) {
            var target = e.target;
            var controlClicked = target.closest('.control-btn');
            var controls;

            if (!controlClicked) {
                return;
            }

            controls = controlClicked.parentElement.querySelectorAll('.control-btn');

            Array.prototype.forEach.call(controls, function (control) {
                if (controlClicked !== control) {
                    control.classList.remove('active');
                } else {
                    controlClicked.classList.toggle('active');
                }
            });

            if (controlClicked.classList.contains('active')) {
                setMode(controlClicked.getAttribute('data-mode'));
            } else {
                disableAllModes()
            }
        });

        function init() {
            forEachMode(function (modeName) {
                var moduleInit = modes[modeName].module.init;

                if (moduleInit) {
                    moduleInit();
                }
            });
        }

        return {
            init: init,
            setMode: setMode
        };
    })();
    // ===========================================================================
    // end MODES MODULE
    // ===========================================================================


    // when map full loaded
    map.on('load', function () {
        canvas = map.getCanvasContainer();

        featuresModule.saveInfo(function() {
            layersModule.init();
            overlayModule.hide();
            // heatmapModule.init();
        });

        stylesModule.init();
        movesModule.init();
        modesModule.init();

        // ======================== EVENT HANDLERS =================================

        // When a click event occurs near a polygon, open a popup at the location of
        // the feature, with description HTML from its properties.
        map.on('click', function (e) {
            var features = map.queryRenderedFeatures(e.point, { layers: layersIds });

            if (!features.length) {
                return;
            }

            featuresModule.openPopup(e.point, features[0]);
        });

        // Use the same approach as above to indicate that the symbols are clickable
        // by changing the cursor style to 'pointer'.
        map.on('mousemove', function (e) {
            var features = map.queryRenderedFeatures(e.point, { layers: layersIds });
            canvas.style.cursor = (features.length) ? 'pointer' : mapSettings.cursor;
        });
    });
})()

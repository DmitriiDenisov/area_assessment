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

    mapboxgl.accessToken = 'pk.eyJ1Ijoib21hbmd1dG92IiwiYSI6ImNpenpvNWtmajAwMjgzMnBmYXp6enhuOTIifQ.KwVxe-y3-a_BeY-uHqBoag';

    var mapRoot = document.getElementById('map-root');
    var mapWrapper = mapRoot.querySelector('.map__wrapper');
    var canvas;

    var mapSettings = {
        cursor: '',
        style: {
            baseURI: 'mapbox://styles/omangutov/',
            defaultStyle: 'cj0tpd64o00j82rnphr76axim'
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
        'buildings-shp': {
            id: 'buildings-shp',
            name: 'Buildings',
            visible: true,
            heatmap: {
                source: 'https://cisdai.ru/buildings-centroids.geojson',
                visible: false
            }
        },
        'circle-farms': {
            id: 'circle-farms',
            visible: true,
            name: 'Circle farms'
        },
        'other-farms': {
            id: 'other-farms',
            visible: true,
            name: 'Other farms'
        }
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
                val = val.toFixed(2);
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
                        '<h3 class="feature__title">Feature info</h3>' +
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
                    (heatmap ?
                    '<div class="layer__control layer__control_full-width">' +
                        '<div class="layer__label">Urbanisation Index</div>' +
                        '<label class="switcher">' +
                            '<input type="checkbox" class="switcher__input layer__heatmap">' +
                            '<span class="switcher__icon"></span>' +
                        '</label>' +
                    '</div>' :
                    '') +
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
                    { name: 'Quantity', value: layer.featuresQuantity },
                    { name: 'Area (m<sup>2</sup>)', value: layer.featuresArea }
                ];

                tmpl +=
                    '<section class="layer" data-id="' + id + '">' +
                        '<h3 class="layer__title">' + layer.name + '</h3>' +
                        renderStats(stats) +
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
                heatmapModule.setVisibility(layerId, heatmapSwitcher.checked);
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
    // end LAYERS MODULE
    // ===========================================================================

    // ===========================================================================
    // ===========================================================================
    // TIPS MODULE
    // ===========================================================================
    // ===========================================================================
    var tipsModule = (function () {
        var tips = document.getElementById('map-tips');

        function setText(text) {
            var content = document.createElement('pre');

            content.classList.add('map-tips__content');
            content.innerHTML = text;
            
            tips.innerHTML = '';
            tips.appendChild(content);
        }

        function clear() {
            var content = tips.querySelector('.map-tips__content');

            if (content) {
                tips.removeChild(content);
            }
        }

        // module exports
        return {
            setText: setText,
            clear: clear
        };
    })();
    // ===========================================================================
    // end TIPS MODULE
    // ===========================================================================


    // ===========================================================================
    // ===========================================================================
    // RULER MODULE
    // ===========================================================================
    // ===========================================================================
    var rulerModule = (function () {
        // GeoJSON object to hold our measurement features
        var geojson = {};
        // Used to draw a line between points
        var linestring = {};
        var display;
        var enabled = false;

        var POINTS_LAYER_ID = 'measure-points';
        var LINE_LAYER_ID = 'measure-lines';

        function resetGeojson() {
            // GeoJSON object to hold our measurement features
            geojson = {
                'type': 'FeatureCollection',
                'features': []
            };

            // Used to draw a line between points
            linestring = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': []
                }
            };
        }

        function updateDisplay(dist) {
            tipsModule.setText('Total distance: ' + dist + 'm');
        }

        function clear() {
            updateDisplay(0);
            resetGeojson();
            map.getSource('geojson').setData(geojson);
        }

        function onClick(e) {
            var features = map.queryRenderedFeatures(e.point, { layers: [POINTS_LAYER_ID] });

            // Remove the linestring from the group
            // So we can redraw it based on the points collection
            if (geojson.features.length > 1) {
                geojson.features.pop();
            }

            // Clear the Distance container to populate it with a new value
            // distanceContainer.innerHTML = '';

            // If a feature was clicked, remove it from the map
            if (features.length) {
                var id = features[0].properties.id;

                geojson.features = geojson.features.filter(function(point) {
                    return point.properties.id !== id;
                });

            } else {
                var point = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [
                            e.lngLat.lng,
                            e.lngLat.lat
                        ]
                    },
                    'properties': {
                        'id': String(new Date().getTime())
                    }
                };

                geojson.features.push(point);
            }

            if (geojson.features.length > 1) {
                linestring.geometry.coordinates = geojson.features.map(function(point) {
                    return point.geometry.coordinates;
                });

                geojson.features.push(linestring);
                updateDisplay(Math.round(turf.lineDistance(linestring) * 1000));
            } else {
                updateDisplay(0);
            }

            map.getSource('geojson').setData(geojson);
        }

        function clearOnEsc(e) {
            if (e.keyCode === 27) {
                clear();
            }
        }

        function addLayers() {
            if (!map.getLayer(POINTS_LAYER_ID)) {
                map.addLayer({
                    id: POINTS_LAYER_ID,
                    type: 'circle',
                    source: 'geojson',
                    paint: {
                        'circle-radius': 5,
                        'circle-color': '#000'
                    },
                    filter: ['in', '$type', 'Point']
                });
            }

            if (!map.getLayer(LINE_LAYER_ID)) {
                map.addLayer({
                    id: LINE_LAYER_ID,
                    type: 'line',
                    source: 'geojson',
                    layout: {
                        'line-cap': 'round',
                        'line-join': 'round'
                    },
                    paint: {
                        'line-color': '#000',
                        'line-width': 2.5
                    },
                    filter: ['in', '$type', 'LineString']
                });
            }
        }

        function removeLayers() {
            if (map.getLayer(POINTS_LAYER_ID)) {
                map.removeLayer(POINTS_LAYER_ID);
            }
            if (map.getLayer(LINE_LAYER_ID)) {
                map.removeLayer(LINE_LAYER_ID);
            }
        }

        function enable() {
            mapSettings.cursor = 'crosshair';

            addLayers();

            map.on('click', onClick);
            document.addEventListener('keydown', clearOnEsc);
        }

        function disable() {
            mapSettings.cursor = '';

            removeLayers();
            clear();

            map.off('click', onClick);
            document.removeEventListener('keydown', clearOnEsc);
        }

        function init() {
            resetGeojson();
            map.addSource('geojson', {
                'type': 'geojson',
                'data': geojson
            });
            addLayers();
        }

        map.on('custom.changestyle', function () {
            init();
        });

        // module exports
        return {
            init: init,
            enable: enable,
            disable: disable
        };
    })();
    // ===========================================================================
    // end RULER MODULE
    // ===========================================================================

    // ===========================================================================
    // ===========================================================================
    // SELECTED AREA MODULE
    // ===========================================================================
    // ===========================================================================
    var selectedAreaModule = (function () {
        var startPoint;
        var currentPoint;
        var selectedBox;
        var mapHelpers = document.getElementById('map-helpers');

        function setFilters(filter) {
            layersIds.forEach(function (id) {
                map.setFilter(id + '-h', filter);
            });
        }

        // Return the xy coordinates of the mouse position
        function getMousePos(e) {
            var rect = canvas.getBoundingClientRect();
            return new mapboxgl.Point(
                e.clientX - rect.left - canvas.clientLeft,
                e.clientY - rect.top - canvas.clientTop
            );
        }
        
        function clear() {
            finish();
            layersModule.removeSelectedArea(mapHelpers);
            setFilters(['in', 'key', '']);
        }

        function finish(bbox) {
            // Remove these events now that finish has been called.
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('keydown', onKeyDown);
            document.removeEventListener('mouseup', onMouseUp);

            if (selectedBox) {
                selectedBox.parentNode.removeChild(selectedBox);
                selectedBox = null;
            }

            // If bbox exists. use this value as the argument for `queryRenderedFeatures`
            if (bbox) {
                var features = map.queryRenderedFeatures(bbox, { layers: layersIds });

                // Run through the selected features and set a filter
                // to match features with unique FIPS codes to activate
                // the `counties-highlighted` layer.
                var filter = features.reduce(function(memo, feature) {
                    if (feature.properties.key) {
                        memo.push(feature.properties.key);
                    }

                    return memo;
                }, ['in', 'key']);

                layersModule.renderSelectedArea(features, mapHelpers);
                setFilters(filter);
            }

            map.boxZoom.enable();
            map.dragPan.enable();
        }

        function onMouseMove(e) {
            // Capture the ongoing xy coordinates
            currentPoint = getMousePos(e);

            // Append the box element if it doesnt exist
            if (!selectedBox) {
                selectedBox = document.createElement('div');
                selectedBox.classList.add('boxdraw');
                canvas.appendChild(selectedBox);
            }

            var minX = Math.min(startPoint.x, currentPoint.x);
            var maxX = Math.max(startPoint.x, currentPoint.x);
            var minY = Math.min(startPoint.y, currentPoint.y);
            var maxY = Math.max(startPoint.y, currentPoint.y);

            // Adjust width and xy position of the box element ongoing
            var pos = 'translate(' + minX + 'px,' + minY + 'px)';

            selectedBox.style.transform = pos;
            selectedBox.style.WebkitTransform = pos;
            selectedBox.style.width = maxX - minX + 'px';
            selectedBox.style.height = maxY - minY + 'px';
        }

        function onMouseUp(e) {
            // Capture xy coordinates
            finish([startPoint, getMousePos(e)]);
        }

        function onKeyDown(e) {
            // If the ESC key is pressed
            if (e.keyCode === 27) {
                finish();
            }
        }

        function clearOnEsc(e) {
            if (e.keyCode === 27) {
                clear();
            }
        }

        function onCanvasMouseDown(e) {
            // Continue the rest of the function if the shiftkey is pressed.
            if (!(e.shiftKey && e.button === 0)) {
                return;
            }

            // Disable default drag zooming when the shift key is held down.
            map.boxZoom.disable();
            map.dragPan.disable();

            // Call functions for the following events
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
            document.addEventListener('keydown', onKeyDown);

            // Capture the first xy coordinates
            startPoint = getMousePos(e);
        }

        function enable() {
            // Set `true` to dispatch the event before other functions
            // call it. This is necessary for disabling the default map
            // dragging behaviour.
            tipsModule.setText('Hold <kbd>shift</kbd> and drag the map to query features area');
            canvas.addEventListener('mousedown', onCanvasMouseDown, true);
            document.addEventListener('keydown', clearOnEsc);
            mapRoot.classList.add('no-select');
        }

        function disable() {
            clear();
            canvas.removeEventListener('mousedown', onCanvasMouseDown, true);
            document.removeEventListener('keydown', clearOnEsc);
            mapRoot.classList.remove('no-select');
        }

        map.on('custom.changestyle', function () {
            clear();
        });

        // module exports
        return {
            enable: enable,
            disable: disable
        };
    })();
    // ===========================================================================
    // end SELECTED AREA MODULE
    // ===========================================================================


    // ===========================================================================
    // ===========================================================================
    // CLUSTERS MODULE
    // ===========================================================================
    // ===========================================================================
    var heatmapModule = (function () {
        // Each point range gets a different fill color.
        var clusters = [
            [0, 'green'],
            [20, 'orange'],
            [200, 'red']
        ];

        function addLayers(source, preffix) {
            var sourceName = preffix + '-heatmap';
            
            map.addSource(sourceName, {
                type: 'geojson',
                data: source,
                cluster: true,
                clusterMaxZoom: 15, // Max zoom to cluster points on
                clusterRadius: 20 // Use small cluster radius for the heatmap look
            });

            clusters.forEach(function (cluster, i) {
                map.addLayer({
                    'id': preffix + '-cluster-' + i,
                    'type': 'circle',
                    'source': sourceName,
                    'paint': {
                        'circle-color': cluster[1],
                        'circle-radius': 70,
                        'circle-blur': 1 // blur the circles to get a heatmap look
                    },
                    'filter': i === clusters.length - 1 ?
                        ['>=', 'point_count', cluster[0]] :
                        ['all',
                            ['>=', 'point_count', cluster[0]],
                            ['<', 'point_count', clusters[i + 1][0]]]
                }, 'waterway-label');
            });
        }
        
        function setVisibility(layerId, visibility) {
            var heatmap = layers[layerId].heatmap;

            if (!heatmap) {
                return;
            }

            heatmap.visible = visibility;

            for (var i = 0; i < 3; i++) {
                var heatmapLayerId = layerId + '-cluster-' + i;
                var visiblityVal = visibility ? 'visible' : 'none';
                
                if (!map.getLayer(heatmapLayerId)) {
                    continue;
                }

                map.setLayoutProperty(heatmapLayerId, 'visibility', visiblityVal);
            }
        }

        function init() {
            layersIds.forEach(function (id) {
                var layer = layers[id];
                var heatmap = layer.heatmap;

                if (!heatmap) {
                    return;
                }

                addLayers(heatmap.source, layer.id);
                setVisibility(id, heatmap.visible);
            });
        }

        map.on('custom.changestyle', function () {
            init();
        });

        // module exports
        return {
            init: init,
            setVisibility: setVisibility
        };
    })();
    // ===========================================================================
    // end CLUSTERS AREA MODULE
    // ===========================================================================


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
            streets: {
                url: 'cj122xjma00312rm0odq908lt',
                name: 'Streets'
            },
            dark: {
                url: 'cj123niht008c2so78qf96zla',
                name: 'Dark'
            }
        };

        function render() {
            var tmpl = '';
            
            for (var prop in styles) {
                if (!styles.hasOwnProperty(prop)) {
                    continue;
                }

                var style = styles[prop];

                tmpl += 
                    '<label class="radio-btn">' +
                        '<input ' +
                            'name="' + INPUT_NAME + '" ' +
                            'type="radio" ' +
                            'class="radio-btn__input" ' +
                            'value="' + prop + '" ' +
                            (style.url === mapSettings.style.defaultStyle ? 'checked' : '') +
                        '>' +
                        '<span class="radio-btn__text">' + style.name + '</span>' +
                    '</label> ';
            }

            stylesWrapper.insertAdjacentHTML('beforeend', tmpl);
        }

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
            render();
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
            ruler: {
                module: rulerModule,
                title: 'Ruler mode'
            },
            selected: {
                module: selectedAreaModule,
                title: 'Selection mode'
            }
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
            tipsModule.clear();
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

            tipsModule.clear();
            activeMode = mode;
            modes[mode].module.enable();
        }

        renderControls();

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
            heatmapModule.init();
        });

        stylesModule.init();
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

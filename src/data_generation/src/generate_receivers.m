function [X_m, X_deg, jammer_loc_m] = generate_receivers(jammer_loc_deg, obs_area, desiredReceivers, buildings_osm, outside_buildings)
    % Generate receiver and jammer positions in meters relative to min latitude and longitude.
    % Returns:
    %   X_meters: Receiver positions in meters
    %   jammer_loc_m: Jammer positions in meters
    %
    % Inputs:
    %   jammer_loc: [T x 2] matrix of jammer positions (latitude, longitude)
    %   obs_area: Scalar defining the area (km^2) of the square around the jammer
    %   desiredReceivers: Number of receiver positions to generate
    %   buildings_osm: Path to the OSM file containing building data

    buildings = readgeotable(buildings_osm, Layer = "buildingparts");
    building_shapes = buildings.Shape; % Geopolyshape array

    [T, D] = size(jammer_loc_deg);
    N = desiredReceivers;
    X_deg = zeros(N, D, T);

    side_length = sqrt(obs_area)*1e-3; % Side length in km

    % Generate valid positions for each time step
    for t = 1:T
        validPositions = nan(N, 2); % Preallocate an Nx2 matrix with NaN values
        
        delta_deg_lat = km2deg(side_length / 2);
        delta_deg_lon = km2deg(side_length / 2 / cosd(jammer_loc_deg(t, 1)));
    
        % Define bounds for the square region around the jammer
        lat_min = jammer_loc_deg(t, 1) - delta_deg_lat;
        lat_max = jammer_loc_deg(t, 1) + delta_deg_lat;
        lon_min = jammer_loc_deg(t, 2) - delta_deg_lon;
        lon_max = jammer_loc_deg(t, 2) + delta_deg_lon;

        % lat_min = jammer_loc_deg(t, 1) - 1/2*delta_deg_lat;
        % lat_max = jammer_loc_deg(t, 1) + 3/2*delta_deg_lat;
        % lon_min = jammer_loc_deg(t, 2) - 1/2*delta_deg_lon;
        % lon_max = jammer_loc_deg(t, 2) + 3/2*delta_deg_lon;
    
        count = 0; % Initialize counter for valid positions
        while count < N
            rxLat = lat_min + (lat_max - lat_min) * rand();
            rxLon = lon_min + (lon_max - lon_min) * rand();
            query_point = geopointshape(rxLat, rxLon);
            
            if outside_buildings
                % Check if the position is outside all buildings
                inside_any_building = false;
                for bldg = 1:length(building_shapes)
                    if isinterior(building_shapes(bldg), query_point)
                        inside_any_building = true;
                        break;
                    end
                end
            end
            
            if (~outside_buildings) || (outside_buildings && ~inside_any_building)
                count = count + 1;
                validPositions(count, :) = [rxLat, rxLon]; % Assign to preallocated array
            end
        end
        X_deg(:, :, t) = validPositions;
    end
    % Combine all receiver and jammer locations to determine min latitude and longitude
    all_latitudes = [X_deg(:, 1, :); jammer_loc_deg(:, 1)];
    all_longitudes = [X_deg(:, 2, :); jammer_loc_deg(:, 2)];
    min_lat = min(all_latitudes(:));
    min_lon = min(all_longitudes(:));

    % Convert all positions to meters relative to (min_lat, min_lon)
    X_m = zeros(N, D, T);
    jammer_loc_m = zeros(T, D);
    cos_min_lat = cosd(min_lat);

    for t = 1:T
        delta_lat = X_deg(:, 1, t) - min_lat;
        delta_lon = X_deg(:, 2, t) - min_lon;

        X_m(:, 1, t) = deg2km(delta_lon * cos_min_lat) * 1000;
        X_m(:, 2, t) = deg2km(delta_lat) * 1000;
    
        % Compute jammer location for time step t
        delta_lat_jammer = jammer_loc_deg(t, 1) - min_lat;
        delta_lon_jammer = jammer_loc_deg(t, 2) - min_lon;
    
        jammer_loc_m(t, 2) = deg2km(delta_lat_jammer) * 1000;
        jammer_loc_m(t, 1) = deg2km(delta_lon_jammer * cos_min_lat) * 1000;
    end
end
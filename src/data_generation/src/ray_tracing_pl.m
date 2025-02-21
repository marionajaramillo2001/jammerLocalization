function losses = ray_tracing_pl(jammer_loc_deg,X_deg,f_jam,P_tx, buildings_osm)
%Computes the path loss using a ray tracing model in an urban scenario

viewer = siteviewer(Buildings= buildings_osm);

% pm = propagationModel("raytracing","Method","sbr",  "MaxNumReflections",4);
pm = propagationModel("raytracing", Method = "sbr", MaxNumReflections = 100, MaxAbsolutePathLoss=160);


[N,D,T] = size(X_deg);
tx_positions = jammer_loc_deg;
rx_positions = X_deg;
losses = zeros(N,T);

for t = 1:T
    tx = txsite("Latitude",tx_positions(t,1), ...
    "Longitude",tx_positions(t,2), ...
    "TransmitterFrequency",f_jam,"TransmitterPower",P_tx);
    for ii = 1:N
        rx = rxsite("Latitude",rx_positions(ii,1,t), ...
        "Longitude",rx_positions(ii,2,t));
        
        % The sigstrength function in MATLAB typically returns the signal
        % strength in dBm (decibels relative to 1 milliwatt), we convert it
        % to dbW by substracting 30
        ss = sigstrength(rx,tx,pm)-30; % signal strength in dBm (accounts for several multipath rays)
        
        losses(ii,t) = P_tx - ss;
    end

end

end


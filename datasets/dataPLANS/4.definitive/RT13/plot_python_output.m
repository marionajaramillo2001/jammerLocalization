% plot python stuffvec_testsvec_tests
close all
clearvars

plot_vs_INRs = 0;


if not(plot_vs_INRs)
    fixed_INR = 5; % chose from: 0	5	10	20	30
%     tests = ["PL1" "PL2" "PL3" "PL4" "PL5"];
%     tests = ["RT1" "RT2" "RT3" "RT4" "RT5"];
%     vec_tests = [1.2e-2 1e-2 0.8e-2 0.6e-2 0.4e-2]; ax_txt = 'obs. density (obs/m$^2$)'; % obs density

%     tests = ["PL2" "PL7" "PL9" "PL10" "PL11"];
    tests = ["RT2" "RT7" "RT9" "RT10" "RT11"];
    vec_tests = [0 1 3 6 10];  ax_txt = '$\sigma_{\mathrm{pos}}$ (m)'; % POs est error variance
    L = length(tests);

    %     vec_tests = 1:L; ax_txt = 'tests (no.)';

    crbx_vec = zeros(1,L);
    crby_vec = zeros(1,L);
    meanx_vec = zeros(1,L);
    meany_vec = zeros(1,L);
    mle_meanx_vec = zeros(1,L);
    mle_meany_vec = zeros(1,L);
    mle_rmsx_vec = zeros(1,L);
    mle_rmsy_vec = zeros(1,L);
    rmsx_vec = zeros(1,L);
    rmsy_vec = zeros(1,L);
    truex_vec = zeros(1,L);
    truey_vec = zeros(1,L);

    for ii = 1:L
%         path = char(strcat('dataPLANS/4.definitive/',tests(ii)));
        path = char(strcat(strcat('dataPLANS/5.NNonly_PLonly/',tests(ii)),'/NNonly'));

        load([path '/matlab/rmsx.mat'])
        load([path '/matlab/rmsy.mat'])
        load([path '/matlab/mle_rmsx.mat'])
        load([path '/matlab/mle_rmsy.mat'])
        load([path '/matlab/crbx.mat'])
        load([path '/matlab/crby.mat'])

        load([path '/matlab/meanx.mat'])
        load([path '/matlab/meany.mat'])
        load([path '/matlab/mle_meanx.mat'])
        load([path '/matlab/mle_meany.mat'])
        load([path '/matlab/truex.mat'])
        load([path '/matlab/truey.mat'])

        load([path '/matlab/snr.mat'])

        idx = find(snr==fixed_INR);

        crbx_vec(ii) = crbx(idx);
        crby_vec(ii) = crby(idx);
        meanx_vec(ii) = meanx(idx);
        meany_vec(ii) = meany(idx);
        mle_meanx_vec(ii) = mle_meanx(idx);
        mle_meany_vec(ii) = mle_meany(idx);
        mle_rmsx_vec(ii) = mle_rmsx(idx);
        mle_rmsy_vec(ii) = mle_rmsy(idx);
        rmsx_vec(ii) = rmsx(idx);
        rmsy_vec(ii) = rmsy(idx);
        truex_vec(ii) = truex(idx);
        truey_vec(ii) = truey(idx);
    end

    % PLOTS
    plot_settings()

    % RMS
    figure
    tiledlayout(2,1)
    nexttile
    plot(vec_tests,crbx_vec,'--k'), hold on, grid on
    plot(vec_tests,mle_rmsx_vec,'color',[.6 .6 .6])
    plot(vec_tests,rmsx_vec,'-k','marker','diamond')
    xlabel(ax_txt)
    ylabel('RMSE$_{\theta_1}$ (m)')
    legend '$\sqrt{CRB}$' 'MLE' 'APBM'

    nexttile
    plot(vec_tests,crby_vec,'--k'), hold on, grid on
    plot(vec_tests,mle_rmsy_vec,'color',[.6 .6 .6])
    plot(vec_tests,rmsy_vec,'-k','marker','diamond')
    xlabel(ax_txt)
    ylabel('RMSE$_{\theta_2}$ (m)')
    legend '$\sqrt{CRB}$' 'MLE' 'APBM'
    savefig(gcf,'figs/matlab_out/RMSE_vs_tests.fig')
    saveas(gcf,'figs/matlab_out/RMSE_vs_tests.eps')

    % Mean
    figure
    tiledlayout(2,1)
    nexttile
    plot(vec_tests,truex_vec,'--k'), hold on, grid on
    plot(vec_tests,mle_meanx_vec,'color',[.6 .6 .6])
    plot(vec_tests,meanx_vec,'-k','marker','diamond')
    xlabel(ax_txt)
    ylabel('$E[{\theta_1}]$ (m)')
    legend 'True' 'MLE' 'APBM'

    nexttile
    plot(vec_tests,truey_vec,'--k'), hold on, grid on
    plot(vec_tests,mle_meany_vec,'color',[.6 .6 .6])
    plot(vec_tests,meany_vec,'-k','marker','diamond')
    xlabel(ax_txt)
    ylabel('$E[{\theta_2}]$ (m)')
    legend 'True' 'MLE' 'APBM'
    savefig(gcf,'figs/matlab_out/mean_vs_tests.fig')
    saveas(gcf,'figs/matlab_out/mean_vs_tests.eps')


else
    load('data/matlab/rmsx.mat')
    load('data/matlab/rmsy.mat')
    load('data/matlab/mle_rmsx.mat')
    load('data/matlab/mle_rmsy.mat')
    load('data/matlab/crbx.mat')
    load('data/matlab/crby.mat')

    load('data/matlab/meanx.mat')
    load('data/matlab/meany.mat')
    load('data/matlab/mle_meanx.mat')
    load('data/matlab/mle_meany.mat')
    load('data/matlab/truex.mat')
    load('data/matlab/truey.mat')

    load('data/matlab/snr.mat')

    plot_settings()

    % RMS
    figure
    tiledlayout(2,1)
    nexttile
    plot(snr,crbx,'--k'), hold on, grid on
    plot(snr,mle_rmsx,'color',[.6 .6 .6])
    plot(snr,rmsx,'-k','marker','diamond')
    xlabel('INR (dB)')
    ylabel('RMSE$_{\theta_1}$ (m)')
    legend '$\sqrt{CRB}$' 'MLE' 'APBM'

    nexttile
    plot(snr,crby,'--k'), hold on, grid on
    plot(snr,mle_rmsy,'color',[.6 .6 .6])
    plot(snr,rmsy,'-k','marker','diamond')
    xlabel('INR (dB)')
    ylabel('RMSE$_{\theta_2}$ (m)')
    legend '$\sqrt{CRB}$' 'MLE' 'APBM'
    savefig(gcf,'figs/matlab_out/RMSE.fig')
    saveas(gcf,'figs/matlab_out/RMSE.eps')

    % Mean
    figure
    tiledlayout(2,1)
    nexttile
    plot(snr,truex,'--k'), hold on, grid on
    plot(snr,mle_meanx,'color',[.6 .6 .6])
    plot(snr,meanx,'-k','marker','diamond')
    xlabel('INR (dB)')
    ylabel('$E[{\theta_1}]$ (m)')
    legend 'True' 'MLE' 'APBM'

    nexttile
    plot(snr,truey,'--k'), hold on, grid on
    plot(snr,mle_meany,'color',[.6 .6 .6])
    plot(snr,meany,'-k','marker','diamond')
    xlabel('INR (dB)')
    ylabel('$E[{\theta_2}]$ (m)')
    legend 'True' 'MLE' 'APBM'
    savefig(gcf,'figs/matlab_out/mean.fig')
    saveas(gcf,'figs/matlab_out/mean.eps')
end



%% Plots

function plot_settings()
defaultAxesFontsize = 16;
defaultLegendFontsize = 16;
defaultLegendInterpreter = 'latex';
defaultLinelinewidth = 2;
defaultStemlinewidth = 2;
defaultAxesTickLabelInterpreter = 'latex';
defaultTextInterpreter = 'latex';
set(0,'defaultAxesFontsize',defaultAxesFontsize,'defaultLegendFontsize',defaultLegendFontsize,...
    'defaultLegendInterpreter',defaultLegendInterpreter,'defaultLinelinewidth',defaultLinelinewidth,...
    'defaultAxesTickLabelInterpreter',defaultAxesTickLabelInterpreter);
set(0,'defaultTextInterpreter',defaultTextInterpreter);
set(0,'defaultFigurePaperPositionMode','auto')
set(0,'defaultStemlinewidth',defaultStemlinewidth)
end
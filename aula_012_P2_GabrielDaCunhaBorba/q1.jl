using Plots
using Dates

# PVI ==========================================================================
# # NOTE - PVI: 2k'' + 3u = 3*cos(3x); u(0) = 0; u'(0)=0;
# function constants_m_k_c()
#     m::Float64 = 2.0;
#     k::Float64 = 3.0;
#     c::Float64 = 0.0;
#     return m,k,c;
# end

# function initialCondition()
#     u::Float64 = 0.0;
#     dudt::Float64 = 0.0;
#     return [u, dudt];
# end

# function p(t)
#     return 3*cos(3*t);
# end

# function f1(t, y1, y2)
#     return y2;
# end

# function f2(t, y1, y2)
#     m, k, c = constants_m_k_c();
#     aux::Float64 = (p(t) - k*y1 - c*y2)/m;
#     return aux;
# end

# function fDifferential(t, y1, y2)
#     vf1::Float64 = f1(t,y1,y2);
#     vf2::Float64 = f2(t,y1,y2);
#     aux = [vf1, vf2];
#     return aux;
# end

# function fAnalytical(t,u,dudt)
#     # LINK - https://www.youtube.com/watch?v=8lPYgL50chI
#     return (u+1/5)*cos(sqrt(3/2)*t) +(sqrt(2/3)*dudt)*sin(sqrt(3/2)*t) -1/5*cos(3*t);
# end

# NOTE - PVI: 1k'' + 1u = 1; u(0) = 0; u'(0)=0;
function constants_m_k_c()
    m::Float64 = 1.0;
    k::Float64 = 1.0;
    c::Float64 = 0.0;
    return m,k,c;
end

function initialCondition()
    u::Float64 = 0.0;
    dudt::Float64 = 0.0;
    return [u, dudt];
end

function p(t)
    return 1;
end

function f1(t, y1, y2)
    return y2;
end

function f2(t, y1, y2)
    m, k, c = constants_m_k_c();
    aux::Float64 = (p(t) - k*y1 - c*y2)/m;
    return aux;
end

function fDifferential(t, y1, y2)
    vf1::Float64 = f1(t,y1,y2);
    vf2::Float64 = f2(t,y1,y2);
    aux = [vf1, vf2];
    return aux;
end

# ==============================================================================
function solvePVI_Analytical(a, b, h, f)
    yi = 0.0;
    ti = a;
    N::UInt16 = trunc((b - a)/h);

    y = zeros(Float64, N+1);
    t = zeros(Float64, N+1);

    u, dudt = initialCondition();

    i::UInt16 = 1;
    while i <= N+1
        # =====   Analytical    =====
        yi = f(ti,u,dudt);
        # ===== =============== =====
        y[i] = yi;
        t[i] = ti;
        ti = a + i * h;
        i += 1;
    end
    return t, y;
end

function aproxPVI_Euler(a, b, h, f, initialCondition)
    N::UInt16 = trunc((b - a)/h);
    lenInitialCondition = length(initialCondition);
    w = Array{Float64, 2}(undef, N+1, lenInitialCondition);
    t = zeros(Float64, N+1);
    wi::Array{Float64, 1} = initialCondition;  # NOTE - equivalent to: wi = y0;
    ti::Float64 = a;

    i::UInt16 = 1;
    while i <= N+1
        selectdim(w, 1, i) .= wi;  # NOTE - equivalent to: w[i] = wi;
        t[i] = ti;
        # =====      Euler      =====
        # wi .= wi .+ h .* f(ti, wi...);
        wi .+= h.*f(ti, wi...);
        # ===== =============== =====
        ti = a + i * h;
        i += 1;
    end
    return t, selectdim(w,2,1);
end

function aproxPVI_RK4(a, b, h, f, initialCondition)
    lenInitialCondition::UInt16 = length(initialCondition);
    N::UInt16 = trunc((b - a)/h);
    w = Array{Float64, 2}(undef, N+1, lenInitialCondition);
    t = zeros(Float64, N+1);
    wi::Array{Float64, 1} = initialCondition;  # NOTE - equivalent to: wi = y0;
    ti::Float64 = a;

    a1::Float64, a2::Float64, a3::Float64, a4::Float64 = 1/6, 1/3, 1/3, 1/6;
    p1::Float64, p2::Float64, p3::Float64 = 1/2, 1/2, 1.0;
    q11::Float64 = 1/2;
    q21::Float64, q22::Float64 = 0, 1/2;
    q31::Float64, q32::Float64, q33::Float64 = 0.0, 0.0, 1.0;

    k1 = Array{Float64, 1}(undef, lenInitialCondition);
    k2 = Array{Float64, 1}(undef, lenInitialCondition);
    k3 = Array{Float64, 1}(undef, lenInitialCondition);
    k4 = Array{Float64, 1}(undef, lenInitialCondition);

    i::UInt16 = 1;
    while i <= N+1
        selectdim(w, 1, i) .= wi;  # NOTE - equivalent to: w[i] = wi;
        t[i] = ti;
        # =====   Runge Kutta   =====
        k1 .= f(ti, wi...);
        # k2 .= f(ti+p1*h, wi + h*(q11*k1));
        # NOTE - the map is applying the anonymous function to each element of wi
        k2 .= f(ti + p1*h, wi .+ h*(q11*k1)...);
        k3 .= f(ti + p2*h, wi .+ h*(q21*k1 + q22*k2)...);
        k4 .= f(ti + p3*h, wi .+ h*(q31*k1 + q32*k2 + q33*k3)...);
        wi .+= h*(a1*k1 + a2*k2 + a3*k3 + a4*k4);
        # ===== =============== =====
        ti = a + i * h;
        i += 1;
    end
    return t, selectdim(w,2,1);
end

function aproxPVI_AB4(a, b, h, f, initialCondition)
    lenInitialCondition::UInt16 = length(initialCondition);
    N::UInt16 = trunc((b - a)/h);
    w = Array{Float64, 2}(undef, N+1, lenInitialCondition);
    t = zeros(Float64, N+1);
    # wi::Array{Float64, 1} = initialCondition;  # NOTE - equivalent to: wi = y0;
    # ti::Float64 = a;

    # NOTE - 4 steps for start the method (the Adams Bashforth method)
    numberSteps = 4;
    tFirstSteps, wFirstSteps = aproxPVI_RK4(a, (numberSteps-1)*h+a, h, f, initialCondition);
    selectdim(w, 1, 1:(numberSteps-1)) .= selectdim(wFirstSteps, 1, 1:numberSteps-1);
    t[1:numberSteps-1] = tFirstSteps[1:numberSteps-1];

    wi = Array{Float64, 1}(undef, lenInitialCondition);
    wi .= selectdim(wFirstSteps, 1, numberSteps);
    ti::Float64 = tFirstSteps[numberSteps];

    i::UInt16 = numberSteps;
    while i <= N+1
        selectdim(w, 1, i) .= wi;  # NOTE - equivalent to: w[i] = wi;
        t[i] = ti;
        # ===== Adams Bashforth =====
        wim1 = selectdim(w,1,i-1);
        wim2 = selectdim(w,1,i-2);
        wim3 = selectdim(w,1,i-3);
        wi .+= (55*f(ti,wi...) - 59*f(t[i-1],wim1...) + 37*f(t[i-2],wim2...) - 9*f(t[i-3],wim3...))*h/24;
        # ===== =============== =====
        ti = a + i * h;
        i += 1;
    end
    return t, selectdim(w,2,1);
end

function aproxPVI_AM4(a, b, h, f, initialCondition)
    lenInitialCondition::UInt16 = length(initialCondition);
    N::UInt16 = trunc((b - a)/h);
    w = Array{Float64, 2}(undef, N+1, lenInitialCondition);
    t = zeros(Float64, N+1);
    # wi::Array{Float64, 1} = initialCondition;  # NOTE - equivalent to: wi = y0;
    # ti::Float64 = a;

    # NOTE - 4 steps for start the method (the Adams Bashforth method)
    numberSteps = 4;
    tFirstSteps, wFirstSteps = aproxPVI_RK4(a, (numberSteps-1)*h+a, h, f, initialCondition);
    selectdim(w, 1, 1:(numberSteps-1)) .= selectdim(wFirstSteps, 1, 1:numberSteps-1);
    t[1:numberSteps-1] = tFirstSteps[1:numberSteps-1];

    wi = Array{Float64, 1}(undef, length(initialCondition));
    wip1 = Array{Float64, 1}(undef, length(initialCondition));  # NOTE - wip1: w_(i+1)
    wi .= selectdim(wFirstSteps, 1, numberSteps);
    ti::Float64 = tFirstSteps[numberSteps];

    i::UInt16 = numberSteps;
    while i <= N+1
        selectdim(w, 1, i) .= wi;  # NOTE - equivalent to: w[i] = wi;
        t[i] = ti;
        # ===== Adams Bashforth =====
        # NOTE - predicting w_(i+1) on t_(i+1) using Adams Bashforth
        wim1 = selectdim(w,1,i-1);  # NOTE - wim1: w_(i-1);
        wim2 = selectdim(w,1,i-2);  # NOTE - wim2: w_(i-2);
        wim3 = selectdim(w,1,i-3);  # NOTE - wim3: w_(i-3);
        wip1 .= wi + (55*f(ti,wi...) - 59*f(t[i-1],wim1...) + 37*f(t[i-2],wim2...) - 9*f(t[i-3],wim3...))*h/24;
        # ===== =============== =====
        # ===== Adams Bashforth =====
        # NOTE - correcting the prediction made before
        tip1 = a + i * h;  # NOTE - tip1: t_(i+1)
        wi .+= (9*f(tip1, wip1...) + 19*f(ti,wi...) - 5*f(t[i-1],wim1...) + f(t[i-2],wim2...))*h/24;
        # ===== =============== =====
        ti = tip1;
        i += 1;
    end
    return t, selectdim(w,2,1);
end

# ==============================================================================
function main()
# SECTION - problem parameters
    #=
    constants for mass: m, elastic constant: k, damping constant: c, should be
    put inside the constants_m_k_c function
    =#

    # NOTE - domainn to use on the methods
    a = 0.0;
    b = 4*(2*π);

    # NOTE - step configuration for configurable plot
    hAnalytical::Float64 = 0.005;
    hEuler::Float64 = 0.1;  # 0.001
    hRK4::Float64   = 0.1;  # 0.1 ok
    hAB4::Float64   = 0.1;
    hAM4::Float64   = 0.1;

    # NOTE - should run the method for the configurable plot
    runAnalyticalConf::Bool = false;  # NOTE - Available only for the test PVI
    runEulerConf::Bool  = true;
    runRK4Conf::Bool    = true;
    runAB4Conf::Bool    = true;
    runAM4Conf::Bool    = true;

    # NOTE - should run the method for the non configurable plot
    runEulerNConf::Bool = false;
    runRK4NConf::Bool   = false;
    runAB4NConf::Bool   = false;
    runAM4NConf::Bool   = false;

    # NOTE - how to output the plot(s)
    output2screen::Bool = true;
    output2image::Bool  = true;

    # NOTE - standardizing the plot(s)
    plotMarkerShape = :circle;
    plotMarkerSize = 2.0;
    plotLineSize = 1.0;
# !SECTION =#

# ==============================================================================
# SECTION - process
    # NOTE - which algorithm to run and plot
    runAnalytical::Bool = runAnalyticalConf;
    runEuler::Bool  = runEulerConf  || runEulerNConf;
    runRK4::Bool    = runRK4Conf    || runRK4NConf;
    runAB4::Bool    = runAB4Conf    || runAB4NConf;
    runAM4::Bool    = runAM4Conf    || runAM4NConf;

    # NOTE - exit if no method should be plotted or no output selected
    if !((runAnalytical || runEuler || runRK4 || runAB4 || runAM4) && (output2screen || output2image))
        println("\n>>> No output selected for the plots...");
        println(">>> Exiting without running the ODE solving methods");
        exit()
    end
    println(">>> Running selected methods...");
    if runAnalytical
        tAnalytical, wAnalytical = solvePVI_Analytical(a, b, hAnalytical, fAnalytical);
    end
    if runEuler
        if runEulerConf
            tEuler, wEuler = aproxPVI_Euler(a, b, hEuler, fDifferential, initialCondition());
        end
        if runEulerNConf
            tEuler_2, wEuler_2 = aproxPVI_Euler(a, b, 0.1, fDifferential, initialCondition());
            tEuler_3, wEuler_3 = aproxPVI_Euler(a, b, 0.05, fDifferential, initialCondition());
            tEuler_4, wEuler_4 = aproxPVI_Euler(a, b, 0.01, fDifferential, initialCondition());
            tEuler_5, wEuler_5 = aproxPVI_Euler(a, b, 0.005, fDifferential, initialCondition());
        end
    end
    if runRK4
        if runRK4Conf
            tRK4, wRK4 = aproxPVI_RK4(a, b, hRK4, fDifferential, initialCondition());
        end
        if runRK4NConf
            tRK4_2, wRK4_2 = aproxPVI_RK4(a, b, 0.1, fDifferential, initialCondition());
            tRK4_3, wRK4_3 = aproxPVI_RK4(a, b, 0.05, fDifferential, initialCondition());
            tRK4_4, wRK4_4 = aproxPVI_RK4(a, b, 0.01, fDifferential, initialCondition());
            tRK4_5, wRK4_5 = aproxPVI_RK4(a, b, 0.005, fDifferential, initialCondition());
        end
    end
    if runAB4
        if runAB4Conf
            tAB4, wAB4 = aproxPVI_AB4(a, b, hAB4, fDifferential, initialCondition());
        end
        if runAB4NConf
            tAB4_2, wAB4_2 = aproxPVI_AB4(a, b, 0.1, fDifferential, initialCondition());
            tAB4_3, wAB4_3 = aproxPVI_AB4(a, b, 0.05, fDifferential, initialCondition());
            tAB4_4, wAB4_4 = aproxPVI_AB4(a, b, 0.01, fDifferential, initialCondition());
            tAB4_5, wAB4_5 = aproxPVI_AB4(a, b, 0.005, fDifferential, initialCondition());
        end
    end
    if runAM4
        if runAM4Conf
            tAM4, wAM4 = aproxPVI_AM4(a, b, hAM4, fDifferential, initialCondition());
        end
        if runAM4NConf
            tAM4_2, wAM4_2 = aproxPVI_AM4(a, b, 0.1, fDifferential, initialCondition());
            tAM4_3, wAM4_3 = aproxPVI_AM4(a, b, 0.05, fDifferential, initialCondition());
            tAM4_4, wAM4_4 = aproxPVI_AM4(a, b, 0.01, fDifferential, initialCondition());
            tAM4_5, wAM4_5 = aproxPVI_AM4(a, b, 0.005, fDifferential, initialCondition());
        end
    end
# !SECTION - process

# ==============================================================================
# SECTION - display
    println(">>> Generating output...");
    plot(title="",
        size=(800,600),
        legend_position=:bottomleft,
        legend_background_color=RGBA(0.0,0.0,0.0,0.0);
    );
    if runAnalytical
        plot!(tAnalytical, wAnalytical,
            label="Analytical",
            linewidth=plotLineSize,
        );
    end
    if runEuler
        if runEulerNConf
            plot!(tEuler_5, wEuler_5,
                label="Euler h:0.005",
                seriestype = :scatter,
                markershape=plotMarkerShape,
                markersize=plotMarkerSize,
                markerstrokewidth=0.001
            );
            plot!(tEuler_4, wEuler_4,
                label="Euler h:0.01",
                seriestype = :scatter,
                markershape=plotMarkerShape,
                markersize=plotMarkerSize,
                markerstrokewidth=0.001
            );
            plot!(tEuler_3, wEuler_3,
                label="Euler h:0.05",
                seriestype = :scatter,
                markershape=plotMarkerShape,
                markersize=plotMarkerSize,
                markerstrokewidth=0.001
            );
            plot!(tEuler_2, wEuler_2,
                label="Euler h:0.1",
                seriestype = :scatter,
                markershape=plotMarkerShape,
                markersize=plotMarkerSize,
                markerstrokewidth=0.001
            );
        end
        if runEulerConf
            plot!(tEuler, wEuler,
                label="Euler h:$hEuler",
                seriestype = :scatter,
                markershape=plotMarkerShape,
                markersize=plotMarkerSize,
                markerstrokewidth=0.001
            );
        end
    end
    if runRK4
        if runRK4NConf
        plot!(tRK4_5, wRK4_5,
            label="Runge-Kutta Ordem 4 h:0.005",
            seriestype = :scatter,
            markershape=plotMarkerShape,
            markersize=plotMarkerSize,
            markerstrokewidth=0.001
        );
        plot!(tRK4_4, wRK4_4,
            label="Runge-Kutta Ordem 4 h:0.01",
            seriestype = :scatter,
            markershape=plotMarkerShape,
            markersize=plotMarkerSize,
            markerstrokewidth=0.001
        );
        plot!(tRK4_3, wRK4_3,
            label="Runge-Kutta Ordem 4 h:0.05",
            seriestype = :scatter,
            markershape=plotMarkerShape,
            markersize=plotMarkerSize,
            markerstrokewidth=0.001
        );
        plot!(tRK4_2, wRK4_2,
            label="Runge-Kutta Ordem 4 h:0.1",
            seriestype = :scatter,
            markershape=plotMarkerShape,
            markersize=plotMarkerSize,
            markerstrokewidth=0.001
        );
        end
        if runRK4Conf
        plot!(tRK4, wRK4,
            label="Runge-Kutta Ordem 4 h:$hRK4",
            seriestype = :scatter,
            markershape=plotMarkerShape,
            markersize=plotMarkerSize,
            markerstrokewidth=0.001
        );
        end
    end
    if runAB4
        if runAB4NConf
            plot!(tAB4_5, wAB4_5,
            label="Adams-Bashforth 4 passos h:0.005",
            seriestype = :scatter,
            markershape=plotMarkerShape,
            markersize=plotMarkerSize,
            markerstrokewidth=0.001
            );
            plot!(tAB4_4, wAB4_4,
            label="Adams-Bashforth 4 passos h:0.01",
            seriestype = :scatter,
            markershape=plotMarkerShape,
            markersize=plotMarkerSize,
            markerstrokewidth=0.001
            );
            plot!(tAB4_3, wAB4_3,
            label="Adams-Bashforth 4 passos h:0.05",
            seriestype = :scatter,
            markershape=plotMarkerShape,
            markersize=plotMarkerSize,
            markerstrokewidth=0.001
            );
            plot!(tAB4_2, wAB4_2,
            label="Adams-Bashforth 4 passos h:0.1",
            seriestype = :scatter,
            markershape=plotMarkerShape,
            markersize=plotMarkerSize,
            markerstrokewidth=0.001
            );
        end
        if runAB4Conf
            plot!(tAB4, wAB4,
            label="Adams-Bashforth 4 passos h:$hAB4",
            seriestype = :scatter,
            markershape=plotMarkerShape,
            markersize=plotMarkerSize,
            markerstrokewidth=0.001
            );
        end
    end
    if runAM4
        if runAM4NConf
            plot!(tAM4_5, wAM4_5,
                label="Adams-Moulton 3 passos h:0.005",
                seriestype = :scatter,
                markershape=plotMarkerShape,
                markersize=plotMarkerSize,
                markerstrokewidth=0.001
            );
            plot!(tAM4_4, wAM4_4,
                label="Adams-Moulton 3 passos h:0.01",
                seriestype = :scatter,
                markershape=plotMarkerShape,
                markersize=plotMarkerSize,
                markerstrokewidth=0.001
            );
            plot!(tAM4_3, wAM4_3,
                label="Adams-Moulton 3 passos h:0.05",
                seriestype = :scatter,
                markershape=plotMarkerShape,
                markersize=plotMarkerSize,
                markerstrokewidth=0.001
            );
            plot!(tAM4_2, wAM4_2,
                label="Adams-Moulton 3 passos h:0.1",
                seriestype = :scatter,
                markershape=plotMarkerShape,
                markersize=plotMarkerSize,
                markerstrokewidth=0.001
            );
        end
        if runAM4Conf
            plot!(tAM4, wAM4,
                label="Adams-Moulton 3 passos h:$hAM4",
                seriestype = :scatter,
                markershape=plotMarkerShape,
                markersize=plotMarkerSize,
                markerstrokewidth=0.001
            );
        end
    end

    if output2image
        date = Dates.format(Dates.now(),"yyyymmddHHMMSS");
        savefig("./output/q1run$date.png");
    end
    # NOTE - screen output must occur after image output
    if output2screen
        plot!(show=true);
        #= NOTE - This prompt prevent any window opened with a plot from being
        automaticaly closed, which occurs when runnig the code directly from
        terminal =#
        print("\n>>> Press <<Enter>> to close the program...");
        readline();
        println(">>> Program closed.");
    end
# !SECTION - display
end

# ==============================================================================
# NOTE - entry point
println(">>> Student: Gabriel da Cunha Borba");
if length(ARGS) != 0
    println(">>> This program doesn't take arguments (yet).");
end
main();
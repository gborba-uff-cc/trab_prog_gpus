# programa que será usado em conunto com o modelador geométrico

#= fluxograma do programa
inicio
1. input >
2. inicializações
3. computar aceleração >
4. computar velocidade >
5. computar deslocamento >
volta ao passo 3 se ainda não terminou o tempo de calculo
6. output
fim
=#

using JSON
using Dates

function readJson(filename)
    open(filename, "r") do fid
        data = JSON.parse(fid);
        key = "step_size";
        if haskey(data, key)
            h = convert(Float64,data[key]);
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "number_steps";
        if haskey(data, key)
            N = convert(UInt32,data[key]);
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        # NOTE - particle properties
        key = "particle_mass";
        if haskey(data, key)
            m = convert(Float64,data[key]);
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "particle_hardness";
        if haskey(data, key)
            k = convert(Float64,data[key]);
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        # key = "particle_radius";
        # if haskey(data, key)
        #     radius = convert(Float64,data[key]);
        # else
        #     println("ERROR: Could not read " * key * " from json");
        #     exit();
        # end
        key = "particle_half_xSize";
        if haskey(data, key)
            dSpringX = convert(Float64,data[key]);
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "particle_half_ySize";
        if haskey(data, key)
            dSpringY = convert(Float64,data[key]);
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "particle_coords";
        if haskey(data, key)
            nE = size(data[key])[1];  # NOTE - number of elements
            x0 = Array{Float64,1}(undef,nE);
            y0 = Array{Float64,1}(undef,nE);
            for i in 1:nE
                x0[i] = convert(Float64,data[key][i][1]);
                y0[i] = convert(Float64,data[key][i][2]);
            end
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "particle_external_force";
        if haskey(data, key)
            # NOTE - number of external forces; x_1, y_1, x_2, y_2, x_3, y_3, ...
            niF = size(data[key], 1);
            njF = size(data[key][1], 1);
            eF = Array{Float64,1}(undef,niF*njF);
            for i in 1:niF
                eF[2*i-1] = convert(Float64,data[key][i][1]);
                eF[2*i]   = convert(Float64,data[key][i][2]);
            end
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "particle_restricted";
        if haskey(data, key)
            # NOTE - number of external forces; x_1, y_1, x_2, y_2, x_3, y_3, ...
            niRestr = size(data[key], 1);
            njRestr = size(data[key][1], 1);
            restr = Array{Float64,1}(undef,niRestr*njRestr);
            for i in 1:niF
                restr[2*i-1] = convert(Float64,data[key][i][1]);
                restr[2*i]   = convert(Float64,data[key][i][2]);
            end
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "particle_connection";
        if haskey(data, key)
            # NOTE - number of external forces; x_1, y_1, x_2, y_2, x_3, y_3, ...
            # conn = convert(Array{UInt16,2}, data[key]);
            # println(size(conn));
            niConn = size(data[key], 1);
            njConn = size(data[key][1], 1);
            conn = Array{Int16,2}(undef,niConn,njConn);
            for i in 1:niConn
                conn[i,1] = convert(Int16,data[key][i][1]);
                conn[i,2] = convert(Int16,data[key][i][2]);
                conn[i,3] = convert(Int16,data[key][i][3]);
                conn[i,4] = convert(Int16,data[key][i][4]);
                conn[i,5] = convert(Int16,data[key][i][5]);
            end
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        # return nE, x0, y0, radius, h, N, m, k, eF, restr, conn
        return nE, x0, y0, dSpringX, dSpringY, h, N, m, k, eF, restr, conn

    end
end

function outputRes(_res, prefix, fileName)
    dict = Dict()
    push!(dict,"resultado"=>_res)
    if length(fileName) == 0
        fileName = dateTimeAsString();
    end
    open(prefix*fileName*".json","w") do f
        JSON.print(f,dict)
    end
end

function dateTimeAsString()
    return Dates.format(Dates.now(),"yyyymmddHHMMSS");
end

function main(filename)
    #= NOTE - read input from json
    nE: number of elements (particles)
    x0, y0: arrays of coordinates
    r : particle radius
    h : step size
    N : number of steps
    m : mass per volume
    k : elastic constant
    f : external forces
    restricted:
    connect   :
    =#
    # nE, x0, y0, r = readJson(filename);
    # nE, x0, y0, r, h, N, m, k, f, restricted, connect = readJson(filename);
    nE, x0, y0, dSpringX, dSpringY, h, N, m, k, f, restricted, connect = readJson(filename);
    # print("nE: ");
    # println(nE);
    # print("x0: ");
    # println(x0);
    # print("y0: ");
    # println(y0);
    # print("r: ");
    # println(r);
    # print("h: ");
    # println(h);
    # print("N: ");
    # println(N);
    # print("m: ");
    # println(m);
    # print("k: ");
    # println(k);
    # print("f: ");
    # println(f);
    # print("restricted: ");
    # println(restricted);
    # print("connect: ");
    # println(connect);
# # ==============================================================================

    # NOTE - initialization
    # num of equations
    nDOFs::UInt32 = nE * 2;  #  2 DOFs per particle (ux, uy)

    # acceleration, velocity, and displacement vectors
    a  = zeros(Float64,nDOFs,1);
    # NOTE - initial conditions enters here
    v  = zeros(Float64,nDOFs,1);
    u  = zeros(Float64,nDOFs,1);
    fi = zeros(Float64,nDOFs,1);

    # NOTE - vetor para guardar historico de deslocamento em um ponto
    u_t = zeros(Float64,N,1);
    # result_index = (nE-5)*2-1;  # NOTE - x of particle n
    result_index = (95)*2-1;  # NOTE - x of particle 95

    # Equation: f_ext - ku = mü; f_ext - fi = mü
    # NOTE - resolving with leapfrog
    a .= (f.-fi)./m;  # calculate initial acceleration; . is to use simd instruction
    for ii in 1:N
        v .+= a.*(0.5*h);  # in the middle of a step
        u .+= v.*h;

        # NOTE - (algoritmo de contato)
        # for better performance use an octree, tree for spacial distributed data
        fi .= 0;
        for jj in 1:nE
            # NOTE - number of neighbors particles
            # NOTE - imposing contour conditions
            u[jj*2-1] *= 1-restricted[jj*2-1];
            u[jj*2]   *= 1-restricted[jj*2];
            xj = x0[jj]+u[jj*2-1];
            yj = y0[jj]+u[jj*2];
            for ww in 1:connect[jj,1]
                neighbor = connect[jj,ww+1];
                xw = x0[neighbor]+u[neighbor*2-1];
                yw = y0[neighbor]+u[neighbor*2];
                dx = xj-xw;
                dy = yj-yw
                d = sqrt(dx^2+dy^2);
                # spring_deformation = d-2*r;
                x_spring_deformation = d-2*dSpringX;
                y_spring_deformation = d-2*dSpringY;
                dx = x_spring_deformation*dx/d;
                dy = y_spring_deformation*dy/d;
                fi[jj*2-1] += k*dx;
                fi[jj*2]   += k*dy;
            end
        end

        # NOTE - store result
        u_t[ii] = u[result_index];

        a .= (f.-fi)./m;
        v .+= a.*(0.5*h);
    end

    outputRes(u_t[:,1],"./output/q2out_","");
end

if (length(ARGS) == 0)
    println("Não foi fornecido o nome do arquivo json de entrada");
elseif (length(ARGS) == 1)
    main(ARGS[1])
end
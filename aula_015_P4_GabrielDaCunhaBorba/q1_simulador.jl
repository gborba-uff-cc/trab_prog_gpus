using JSON
using SparseArrays
using Dates

#=
SECTION - entrada
- considerando espaçamento regular em x e y
    espaçamento em x := h, espaçamento em k := k
- condição de contorno
- informações de vizinhança
    indices de vizinhaça:
      4
    2 0 1
      3


!SECTION
=#


function readJson(filename)
    open(filename, "r") do fid
        data = JSON.parse(fid);

        key = "x_dist";
        if haskey(data, key)
            h = convert(Float64,data[key]);
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "y_dist";
        if haskey(data, key)
            k = convert(Float64,data[key]);
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "ij_pos";
        if haskey(data, key)
            niGridPos = size(data[key], 1);
            njGridPos = size(data[key][1], 1);
            gridPos = Array{Int16,2}(undef,niGridPos,njGridPos);
            for i in 1:niGridPos
                for j in 1:njGridPos
                    gridPos[i,j] = convert(Int16,data[key][i][j]);
                end
            end
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "connect";
        if haskey(data, key)
            niConn = size(data[key], 1);
            njConn = size(data[key][1], 1);
            conn = Array{Int16,2}(undef,niConn,njConn);
            for i in 1:niConn
                for j in 1:njConn
                    conn[i,j] = convert(Int16,data[key][i][j]);
                end
            end
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end
        key = "boundary_condition"
        if haskey(data, key)
            niBc = size(data[key], 1);
            njBc = size(data[key][1], 1);
            bc = Array{Int16,2}(undef,niBc,njBc);
            for i in 1:niBc
                for j in 1:njBc
                    bc[i,j] = convert(Int16,data[key][i][j]);
                end
            end
        else
            println("ERROR: Could not read " * key * " from json");
            exit();
        end

        return h,k,conn,bc,gridPos;
    end
end

function outputResJson(x, y, prefix, fileName)  # FIXME
    dict = Dict();
    push!(dict,"domain"=>x);
    push!(dict,"image"=>y);
    if length(fileName) == 0
        fileName = dateTimeAsString();
    end
    open(prefix*fileName*".json","w") do f
        JSON.print(f,dict);
    end
    return;
end

function outputResCsv(x, y, prefix, fileName)
    nE::UInt16 = length(y);
    i::UInt16 = 1;
    open(prefix*fileName*".csv","w") do f
        println(f,"iPos,jPos,Temperature");
        while i <= nE
            println(f, x[i,1],',',x[i,2],',',y[i]);
            i += 1;
        end
    end
    return
end

# ==============================================================================
"""
Get the coeficients that build the A matrix to solve the bidimensional PVC of
temperature.
Returns an array with coeficients for (i,j),(i+1,j),(i-1,j),(i,j-1),(i,j+1)
"""
function getBlock(h,k)
    kC::Float64 = 2*((h/k)^2+1);
    kR::Float64 = -1.0;
    kL::Float64 = kR;
    kB::Float64 = -1*((h/k)^2);
    kT::Float64 = kB;
    return [ kC, kR, kL, kB, kT ];
end
# ==============================================================================

function main(inJsonFilename)
    println(">>> Reading input...");

    h,k,connect,bc,gridPos = readJson(inJsonFilename);
    outPath = Base.Filesystem.dirname(inJsonFilename)*'/';
    inFilename = Base.Filesystem.basename(inJsonFilename);
    outFilename = replace(inFilename, r"(.*)in(\d*)(\..*$)"i=>s"\1Out\2");
    # outNumber = match(r"(\d*)$",outFilename).match;
    if inFilename == outFilename
        now = Dates.now();
        outFilename = Dates.format(now,dateformat"yyyy\ymm\mdd\dHHhMM\mSS\s");
    end

    println(">>> Input read...");
    # --------------------------------------------------------------------------

    println(">>> Starting processing...");
    # NOTE - this block refers to a differential equation that reigns a thermal problem
    block = getBlock(h,k);

    # NOTE - number of points
    nLines, nNeighbours  = size(connect)
    # NOTE- init A and b
    A = spzeros(Float64,nLines,nLines);
    b = zeros(Float64,nLines,1);

    # NOTE - build linear system of equations
    for i = 1:nLines
        println("i: ",i);
        A[i,i] = block[1];
        # NOTE - for each neighbour
        for j = 1:nNeighbours
            # NOTE - loc is one at a moment (right,left,bottom,up) neighbour
            println("connect[", i, ',', j, "]: ", connect[i,j]);
            loc = connect[i,j];
            # NOTE - include in A if there is a neighbour and it isnt a Dirichlet condition
            if loc > 0  # is a neighbour
                # NOTE - neighbour has a Dirichlet condition
                if bc[loc,1] == 1
                    b[i] += bc[loc,2];
                # NOTE - hasn't Dirichlet condition
                else
                    A[i,loc] = block[j+1];
                end
            end
        end
    end

    # NOTE - impose boundary conditions
    for i = 1:nLines
        if bc[i,1] == 1  # NOTE - element has Dirichlet condition
            A[i,:] = zeros(Float64,1,nLines);
            A[i,i] = 1.0;
            b[i,1] = bc[i,2];
        end
    end

    result = A \ b
    println(">>> Processing finished...");
    # --------------------------------------------------------------------------

    println(">>> Generating output...");

    outputResJson(gridPos,result[:,1],outPath,outFilename);
    outputResCsv(gridPos,result[:,1],outPath,outFilename)

    println(">>> Output generated...");
end

println(">>> Student: Gabriel da Cunha Borba\n");
if length(ARGS) == 1
    main(ARGS[1]);
else
    println(">>> This program take 1 argument.");
    println(">>> Run again passing the path to an archive .json containing values to:\n",
    "    <x_dist>, <y_dist>, <ij_pos>, <connect>, <boundary_condiditon>");
end
println(">>> Program closed.");
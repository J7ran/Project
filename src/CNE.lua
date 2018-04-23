-- Memory addresses - https://github.com/rodamaral/smw-tas
-- Inspired by SethBling's MarI/O - https://pastebin.com/ZZmSNaHX
-- Bizhawk Lua Functions - http://tasvideos.org/Bizhawk/LuaFunctions.html

local SMW = require("smw")

-- Environment constants
FileName = "Save.State"
Buttons = {"A", "B", "X", "Y", "Up", "Down", "Left", "Right"}
InputSize = 170
OutputSize = #Buttons
MaxNodes = 1000000
TimeOut = 20

-- CNE constants
PopulationSize = 150
MutationRate = 0.01
MutationSize = 0.2

------------------------------------- ENVIRONMENT FUNCTIONS ------------------------------------
-- Get player position
function getPosition()
	playerX = memory.read_s16_le(SMW.WRAM.x)
	playerY = memory.read_s16_le(SMW.WRAM.y)
end

-- Search for sprites/extended sprites
function getSprites(spriteType)
	local sprites = {}
	-- Search through all sprites and keep track of interactable (alive) sprites
	if spriteType == "sprite" then
		for i = 0, SMW.constant.sprite_max-1 do
			local status = memory.readbyte(SMW.WRAM.sprite_status+i)
			-- If alive
			if status ~= 0 then
				local x = memory.readbyte(SMW.WRAM.sprite_x_low+i) 
							+ memory.readbyte(SMW.WRAM.sprite_x_high+i)*256
				local y = memory.readbyte(SMW.WRAM.sprite_y_low+i) 
							+ memory.readbyte(SMW.WRAM.sprite_y_high+i)*256
				sprites[#sprites+1] = {["x"] = x, ["y"] = y}
			end
		end
	-- Search through all extended sprites and keep track of interactable (alive) ones
	elseif spriteType == "extended" then
		for i = 0, SMW.constant.extended_sprite_max-1 do
			local status = memory.readbyte(SMW.WRAM.extspr_number+i)
			-- If alive
			if status ~= 0 then
				local x = memory.readbyte(SMW.WRAM.extspr_x_low+i) 
							+ memory.readbyte(SMW.WRAM.extspr_x_high+i)*256
				local y = memory.readbyte(SMW.WRAM.extspr_y_low+i) 
							+ memory.readbyte(SMW.WRAM.extspr_y_high+i)*256
				sprites[#sprites+1] = {["x"] = x, ["y"] = y}
			end
		end
	end
	
	return sprites
end

-- Get the tile with respect to player's position from Map16 low byte table
function getTiles(dx, dy)
	local x = math.floor((playerX + dx + 8) / 16)
	local y = math.floor((playerY + dy) / 16)
	
	local tile = SMW.constant.map16_low + 
					math.floor(x/16) * SMW.constant.room_size + y*16 + x%16
	
	return memory.readbyte(tile)
end

-- Get input tiles: Empty tiles, Ground tiles, Object tiles
function getInputs()
	getPosition()
	local sprites = getSprites("sprite")
	local extendedSprites = getSprites("extended")
	
	local inputs = {}
	-- A tile is 16 bytes
	-- Determine the tiles from 6 down to 6 up
	for tileY = -96, 96, 16 do
		-- Determine the tiles from 6 left to 6 right
		for tileX = -96, 96, 16 do
		
			-- Initialise to empty tile
			inputs[#inputs+1] = 0
			
			-- If tile is a ground tile, set to 1
			local tile = getTiles(tileX, tileY)
			if tile == 1 and playerY+tileY < SMW.constant.room_size then
				inputs[#inputs] = 1
			end
			
			-- If object tile (enemies and power-ups), set to -1
			for i = 1, #sprites do
				local x = math.abs(sprites[i]["x"] - (playerX+tileX))
				local y = math.abs(sprites[i]["y"] - (playerY+tileY))
				if x <= 8 and y <= 8 then
					inputs[#inputs] = -1
				end
			end
			for i = 1, #extendedSprites do
				local x = math.abs(extendedSprites[i]["x"] - (playerX+tileX))
				local y = math.abs(extendedSprites[i]["y"] - (playerY+tileY))
				if x <= 8 and y <= 8 then
					inputs[#inputs] = -1
				end
			end
		end
	end
	
	return inputs
end

-- Sigmoid for activation function
function sigmoid(x)
    return 2 / (1 + math.exp(-4.9*x)) - 1
end

-- Clear the controller
function clearJoypad()
    controller = {}
	
    for i = 1, #Buttons do
		controller["P1 " .. Buttons[i]] = false
    end
	
    joypad.set(controller)
end

-- Check for invalid button combinations
function checkJoypad()
	if controller["P1 Left"] and controller["P1 Right"] then
		controller["P1 Left"] = false
		controller["P1 Right"] = false
	end
	if controller["P1 Up"] and controller["P1 Down"] then
		controller["P1 Up"] = false
		controller["P1 Down"] = false
	end
	
	joypad.set(controller)
end

-- Draw GUI
function GUI()
	gui.drawBox(5, 9, 250, 74, 0xFF2D313C, 0xFF26CDCD)
	
	gui.drawText(15, 15, "Training an agent using Neuroevolution", 0xFFFFFFFF, 11, "Arial", "Bold")
	gui.drawText(18, 37, "Generation: " .. population.generation, 0xFFFFFFFF, 11, 11, "Arial")
	gui.drawText(95, 37, "Max Fitness: " .. population.maxFitness, 0xFFFFFFFF, 11, 11, "Arial")
	gui.drawText(18, 57, "Genome: " .. population.currentGenome, 0xFFFFFFFF, 11, 11, "Arial")
	gui.drawText(95, 57, "Fitness: " .. math.floor(progress - population.frame/2 - (timer + population.frame/4)*2/3), 0xFFFFFFFF, 11, 11, "Arial")
	gui.drawText(216, 47, math.floor(population.currentGenome/PopulationSize*100) .. "%", 0xFFFFFFFF, 11, 12, "Arial")

	gui.drawLine(5, 34, 250, 34, 0xFF2D313C)
	gui.drawLine(5, 54, 210, 54, 0xFF2D313C)
	gui.drawLine(210, 34, 210, 74, 0xFF2D313C)
end
------------------------------------------------------------------------------------------------

----------------------------------- NEURAL NETWORK FUNCTIONS -----------------------------------
-- Create a population 
function createPopulation()
	local population = {}
	
	population.species = {}
	population.generation = 1
	
	-- Pointers to current species and genome
	population.currentSpecies = 1
	population.currentGenome = 1
	
	-- Pointers to current frame and max fitness of the population
	population.frame = 0
	population.maxFitness = 0
	
	return population
end

-- Create a species
function createSpecies()
	local species = {}
	species.genomes = {}
	
	return species
end

-- Create a genome
function createGenome()
	local genome = {}
	
	genome.genes = {}
	genome.fitness = 0
	genome.probability = 0.0
	genome.network = {}
	
	return genome
end

-- Create a gene
function createGene()
	local gene = {}

	gene.inc = 0
	gene.out = 0
	gene.weight = 0.0
	gene.enabled = true
	
	return gene
end

-- Copy a gene
function copyGene(gene)
    local gene2 = createGene()
	
    gene2.inc = gene.inc
    gene2.out = gene.out
    gene2.weight = gene.weight
    gene2.enabled = gene.enabled
    
    return gene2
end

-- Create a link between an input and non-input neuron
function createLink(genome)
	local inputNeuron = randomNeuron(genome.genes, false)
	local nonInputNeuron = randomNeuron(genome.genes, true)
	
	local link = createGene()
	link.inc = inputNeuron
	link.out = nonInputNeuron
	link.weight = math.random()*2 - 1
	
	for i = 1, #genome.genes do
		local gene = genome.genes[i]
		if gene.inc == link.inc and gene.out == link.out then
			return
		end
	end
	
	table.insert(genome.genes, link)
end

-- Add a bias link
function addBias(genome)
	if math.random(2) == 1 then
		local neuron = randomNeuron(genome.genes, true)
		
		local link = createGene()
		link.inc = InputSize
		link.out = neuron
		link.weight = math.random()*2 - 1
		
		for i = 1, #genome.genes do
			local gene = genome.genes[i]
			if gene.inc == link.inc and gene.out == link.out then
				return
			end
		end
		
		table.insert(genome.genes, link)
	end
end

-- Create a neuron
function createNeuron()
	local neuron = {}
	
	neuron.incoming = {}
	neuron.inputValue = 0.0
	
	return neuron
end

-- Return a random input or non-input neuron
function randomNeuron(genes, nonInput)
    local neurons = {}
	
    if not nonInput then
        for i = 1, InputSize do
            neurons[i] = true
        end
    end
	
    for o = 1, OutputSize do
        neurons[MaxNodes+o] = true
    end
	
    for i = 1, #genes do
        if (not nonInput) or genes[i].inc > InputSize then
            neurons[genes[i].inc] = true
        end
        if (not nonInput) or genes[i].out > InputSize then
            neurons[genes[i].out] = true
        end
    end
 
    local count = 0
    for _,_ in pairs(neurons) do
        count = count + 1
    end
    local n = math.random(1, count)
    
    for k,v in pairs(neurons) do
        n = n-1
        if n == 0 then
            return k
        end
    end
    
    return 0
end

-- Create the Neural Network from genome
function createNetwork(genome)
	local network = {}
	network.neurons = {}
	
	for i = 1, InputSize do
		network.neurons[i] = createNeuron()
	end
	
	for i = 1, OutputSize do
		network.neurons[MaxNodes+i] = createNeuron()
	end
	
	table.sort(genome.genes, function (a,b) return (a.out < b.out) end)
	
	for i = 1, #genome.genes do
		local gene = genome.genes[i]
		if gene.enabled then
			if network.neurons[gene.out] == nil then
				network.neurons[gene.out] = createNeuron()
			end
			
			local neuron = network.neurons[gene.out]
			table.insert(neuron.incoming, gene)
			
			if network.neurons[gene.inc] == nil then
				network.neurons[gene.inc] = createNeuron()
			end
		end
	end
	
	genome.network = network
end

-- Get the outputs for the given network
function evaluateNetwork(network)	
	table.insert(inputs, 1)

	for i = 1, InputSize do
		network.neurons[i].inputValue = inputs[i]
	end
	
	-- Activation function
	for _,neuron in pairs(network.neurons) do
		local sum = 0
		for i = 1, #neuron.incoming do
			local incoming = neuron.incoming[i]
			local other = network.neurons[incoming.inc]
			sum = sum + incoming.weight * other.inputValue
		end
		if #neuron.incoming > 0 then
			neuron.inputValue = sigmoid(sum)
		end
	end
	
	-- Set the controller using outputs
	local outputs = {}
	for i = 1, OutputSize do
		local button = "P1 " .. Buttons[i]
		if network.neurons[MaxNodes+i].inputValue > 0 then
			outputs[button] = true
		else
			outputs[button] = false
		end
	end
	
	return outputs
end
------------------------------------------------------------------------------------------------

---------------------------------- GENETIC ALGORITHM FUNCTIONS ---------------------------------
-- Biased weight mutation
function mutateWeight(genome)
	for i = 1, #genome.genes do
		local gene = genome.genes[i]
		if math.random() < MutationRate then
			gene.weight = gene.weight + math.random() * MutationSize
		end
	end
end

-- Select two non-input nodes then for each ingoing link, mutate the weights
function mutateNode(genome)
	local n1 = randomNeuron(genome.genes, true)
	local n2 = randomNeuron(genome.genes, true)
	
	for i = 1, #genome.genes do
		local gene = genome.genes[i]
		if gene.out == n1 and math.random() < MutationRate then
			gene.weight = gene.weight + math.random() * MutationSize
		end
		if gene.out == n2 and math.random() < MutationRate then
			gene.weight = gene.weight + math.random() * MutationSize
		end
	end
end

function Mutate(genome)
	mutateWeight(genome)
	mutateNode(genome)
end

-- Set the roulette wheel probabilities for each genome in a species
function setProbabilities(species)
	local sum = 0
	for i = 1, #species.genomes do
		local genome = species.genomes[i]
		sum = sum + genome.fitness
	end
	
	local previous = 0.0
	for i = 1, #species.genomes do
		local genome = species.genomes[i]
		genome.probability = previous + (genome.fitness / sum)
		previous = genome.probability
	end
end

-- Roulette Wheel that returns a genome for reproduction
function rouletteWheel(species)
	local spin = math.random()
	local previous = 0.0
	
	for i = 1, #species.genomes do
		local genome = species.genomes[i]
		previous = previous + genome.probability
		if spin <= previous then
			return genome
		end
	end
end

-- Crossover two genomes to produce a child genome
function Crossover(g1, g2)
	local child = createGenome()
	
	for i = 1, #g1.genes do
		if math.random(2) == 1 and g2.genes[i] ~= nil and g2.enabled then
			table.insert(child.genes, copyGene(g2.genes[i]))
		else
			table.insert(child.genes, copyGene(g1.genes[i]))
		end
	end
	
	return child
end

-- Produce a child genome and mutate
function Reproduce(species)
	local child = {}
	
	setProbabilities(species)
	local g1 = rouletteWheel(species)
	local g2 = rouletteWheel(species)
	child = Crossover(g1, g2)
	
	createLink(child)
	addBias(child)
	Mutate(child)
	
	return child
end

-- Reproduce a new species and start the next generation
function Evolve()	
	local children = createSpecies()
	local species = population.species[population.currentSpecies]
	
	for i = 1, PopulationSize do
		table.insert(children.genomes, Reproduce(species))
	end
	
	population.species[population.currentSpecies] = children
	
	population.generation = population.generation + 1
end

-- Execute a training run for the current genome
function Train()
	clearJoypad()
	savestate.load(FileName)
	
	population.frame = 0
	progress = 0
	timer = TimeOut
	
	local species = population.species[population.currentSpecies]
	local genome = species.genomes[population.currentGenome]
	createNetwork(genome)
	Evaluate()
end

-- Evaluate the current genome
function Evaluate()
	local species = population.species[population.currentSpecies]
	local genome = species.genomes[population.currentGenome]
	
	inputs = getInputs()
	controller = evaluateNetwork(genome.network)
	checkJoypad()
end

-- Go to next genome. If current generation finished, create the next generation
function nextGenome()
	population.currentGenome = population.currentGenome + 1
	if population.currentGenome > #population.species[population.currentSpecies].genomes then
		population.currentGenome = 1
		Evolve()
	end
end

-- Boolean check for if the current genome has been trained
function isTrained()
	local species = population.species[population.currentSpecies]
	local genome = species.genomes[population.currentGenome]
	
	return genome.fitness ~= 0
end
------------------------------------------------------------------------------------------------

---------------------------------------- MAIN EXECUTION ----------------------------------------
if population == nil then
	population = createPopulation()
	
	local newSpecies = createSpecies()
	for i = 1, PopulationSize do
		local genome = createGenome()
		createLink(genome)
		addBias(genome)
		table.insert(newSpecies.genomes, genome)
	end
	table.insert(population.species, newSpecies)
	
	Train()
end

while true do
	local species = population.species[population.currentSpecies]
	local genome = species.genomes[population.currentGenome]
	
	if population.frame%5 == 0 then
		Evaluate()
	end
	joypad.set(controller)
	
	getPosition()
	if playerX > progress then
		progress = playerX
		timer = TimeOut
	end
	
	timer = timer - 1
	if timer + population.frame/4 <= 0 then
		local fitness = progress - population.frame/2
		if progress > 4816 then
			fitness = fitness + 1000
			console.writeline(population.generation)
		end
		genome.fitness = fitness
		
		if fitness > population.maxFitness then
			population.maxFitness = fitness
		end
		
		population.currentGenome = 1
		while isTrained() do
			nextGenome()
		end
		
		Train()
	end
	
	GUI()
	population.frame = population.frame + 1

	emu.frameadvance()
end
''' The complete pipeline for NVDLA from :
	
	DRAM->SRAM->NVDLA->SRAM->DRAM 

	The overall pipeline is split into 3 separate pipeline with stop points at CBUF and CACC-AssemblyGroup 

	First the data(input + weight) is transferred from DRAM to SRAM completely ( input data size is 0.5 MB and weight data size is  )

	(how much data gets transferred depends on several factors such as size of input data (input_size) , size of weight data (weight_size), SRAM size(sram_size))
	Once data is transferred to SRAM : 1) How much data should be transferred to SRAM for the purpose of computation ?
									   2) What is it dependent on ?
									   3) How will it affect the pipeline down the line ?

	From SRAM data will be transferred to CBUF 1) How much data should be transferred ?
											   2) What is this dependent on ?
											   3) How will it affect the pipeline down the line ?

	Once the requisite amount of data has been transferred to CBUF, we can start computing on that data one by one, the atomic operations can be computed 

	This process of computing the atomic operations will be one pipeline: The data leaves CBUF, is transferred to CSC and subsequently computed in CMAC and finally 
	this will be stored in CACC Assembly group. The Assembly group will fill up until all the atomic operations required to compute a channel operation is completed

	At this point the assembly group is full, and the computation has halted until the data from Assembly group is transferred to Delivery group 

	Once all the data has arrived at Delivery group after some amount of time we can again resume computation and at the same time begin transfer of output data from Delivery
	group to SDP. From SDP the computed values can be stored back in the SRAM. This is the third and final pipeline 
		1) WHile the third pipeline is active, what else can be done ? 
		We should keep check of the state of the sytem once the checkpoints are reached.. This will let us know what is happening as the system computes 

		2) So, while third pipeline is active, is there enough data in CBUF for the next set of atomic operations to begin ?
		3) Or do we need to reload CBUF with new set of inputs parallel to the third pipeline ? 
		4) The output of SDP to SRAM will arrive, however is there enough space to keep it ? because for after the first convolution is complete the output data cube generated 
		will have some size and at the same time the SRAM is still consisting of input data which hasn't been computed yet ? 
		Maybe it is better to introduce another stage of SRAM that will contain just the output coming from SDP. This SRAM will be one stage in pipeline behind the main SRAM connecting to 
		CBUF. 

		Now, we need to be careful of how much time it takes to empty Delivery SRAM in CACC versus the time taken to fill Assembly SRAM in CACC. If delivery takes longer to empty 
		then the new channel data output from Assembly will have to wait. Otherwise, if assembly takes longer to fill up, the pipeline 3 would have complete already and we need to just add the 
		time taken for assembly to fill up plus the time taken to transfer data from assembly to delivery.

	All of the above discussion would the pipeline of pipeline 2 + pipeline 3 will have to decided based on the size of data for next channel operation going into CMAC while the 
	previous channel operation data is being transferred in pipeline 3 

	Now we have sort of linked pipeline 2 and 3 into a larger pipeline of the system, we still need to figure out the sizes of data being transferred (the data flow so to speak) within 
	these two pipelines, the final pipeline 1 is yet to be linked into an even higher pipeline which connects the outside storage units to the entire process. 

	As is evident, each input data cube will not be computed all at once, it will be split into smaller subunits and computed in smaller chunks. Which accumulate over time to give us the final output 
	data cube of first layer in SRAM finally, Now this is the pipeline_2_3. That will repeat over time. While the outputs are being generated the weights will also be required to be moved from 
	DRAM to SRAM plus the additional problem of RESNET layers which hasn't been solved yet. 

	This process of weights movement will need to occur such that they are always available for use for the right input values. Even the input values would have to be moved from SRAM to CBUF and so on 

	This part is slightly tricky to handle: What could be the reason ? 

	The data in CBUF after the first transfer has to computed completely before the next set of inputs are added in. So while we are emptying the CBUF we will also be storing the output from this CBUF input 
	back into SRAM. However, there is already the input of the previous layer that we haven't taken in, so we need to compute this first and only then move to the ouput data. So when the CBUF sends empty signal we 
	we ask SRAM to load in the other partial input data and the associated weights into CBUF. This cycle would repeat for the given layer until we exaust all the inputs of this layer present in SRAM completely

	Also, an important note , everytime an input data cube is moved from SRAM to CBUF, we make some space avialable for output of this input cube plus some additional size can be provided in SRAM 
	to prevent overflow into DRAM. 
		Even if there is some overflow into DRAM from the SRAM in case of larger output it should not be a problem, however if the overflow occurs later the better and not too early

		The ouput coming from SDP into SRAM should check for what ?? if there is space avialable in SRAM? Which there should be yes ? Beacuse the data currently being executed is 
		present in CBUF. For any layer, SRAM had 0.5 MB as input, then some of it was transferreed to CBUF, then later some came back from SDP. At the point, the next set of input 
		was already(?) transferred to CBUF based on the pipeline that was mentioned above. 

		So, while delivery SRAM is emptying, the next set of input is being transferred from SRAM - > CBUF or is some data already present in CBUF ?? Check for this
			Because if it is already present in CBUF, then we can simply start transfer from CBUF - > CSC - > CMAC -> Assembly group 
			else if data is not there is CBUF, then we must transfer the requisite data from SRAM - > CBUF to fill it up first ( this case is possibly worse than previous one as CMAC would remain idle during SRAM -> CBUF transfer)

		Now, when the final set of input data has been transferred into CBUF (in case required ) we can be sure than the build of output data now in SRAM over the period can begin feeding into 
		CBUF. This means , that the input data for first layer has been computed and its time to take in new set of weights into SRAM too. 
		Here, since the same SRAM while in between computation of any layer would have a mix of input, output of this input and weight data we need to check if all the weights can be stored 
		in SRAM all at a time or not. At the end of the last input data for current layer we would have emptied SRAM completely and all that would remain is output that was generated from these input + the output yet to arrive
		so the space would only have output cube at this point after final input data has been computed. This should prompt us to do either of two things 
		check for the space left in SRAM and the size of weights for the next layer, 
			if enough space for weights is available send the signal to transfer all weights from DRAM -> SRAM. We also need to keep track of when to send the signal for weights to be transferred from 
			DRAM -> SRAM as this would depend on the size of weights to be transferred

			else if enough space is not available we need to keep split this layer computation in multiple computation with different kernel sets being computed while the input remains in SRAM
				Once split it is just like any layer computation only difference being that the output being generated will have to stored in SRAM as usual but the input data will have to be refresed when the next 
				set of kernels arrive from DRAM. So this part is a little confusing at the moment ? How would we refresh while at the same time store the output being generated ? Because as usual 
				the input data will continue to be consumed for the computation and this means we are left with no input data at the end. But we need it beacuse not all the weights have been computed 
				so finding a way to either cache input data or copy to DRAM could be required. This copying can be done while the output data from the previous computation is being written back to SRAM (as we know), but do the same for DRAM also and copy it there
				just in case a situation like this arises.

							As a side note , this could also solve the problem of Resnet. As input data is 

					Now, when the input data cube has been for next layer will be computed one kernel set a time. When a kernel set is finished we will transfer the same input cube along with the second kernel set
					from DRAM to SRAM as if we were doing this process from the beginning and the whole process continues until all the sets are exausted for this layer
			Now once the last of input cube is computed we would have removed it from SRAM and moved to CBUF. All that will be left in SRAM is the output of this layer which means we should be ready to input the next kernel weight from DRAM
			The above case could arise again and we should repeat this process just described above. Hopefully this is correct at least at this high level (?)


	now for the Resnet issue, every other layer of convolution must cache the output cube for addition later in SDP. I am given to understand that SDP holds buffers for this purpose.
	Based on our discussion above we would copy the output cube in DRAM so that a copy of output cube is always available there. When the Resnet layer is required the copy from DRAM can be transferred. This solves the 
	output data cube caching problem. However, there is still the issue of when to transfer and how much to transfer? How should this be synchronised? Probably a channel operation at a time amount of cached values should be aviable 
	at SDP so we can add this instantly as soon as the output arrives from Delivery group. Just check this? The MCIF can arbitrate this process between DRAM -> SDP and keep the buffer avialable in SDP filled for Resnet activity

	Now is anything else happening while Resnet is supposed to work? Probably the copying is going on into DRAM ?? There could a collision problem which would mean that certain task takes priority and that would be DRAM -> SDP transfer obviously
	Remember we are copying the value of the current input being processed by compute module however the resnet copy of the layer that came two layers in advance must remain cache. This can only be destroyed when the complete resnet procedure is complete
	Since we are talking DRAM to SDP transfer it will be slower but we should ensure that there is enough time for a channel operation worth of resnet data avaibale in SDP buffer when the delivery start to unload into SDP 
	Once the resnet layer has been incorporated into the SDP computation we will proceed as usual. The same steps as described above should follow with the addition that the DRAM -> SDP must also be arbitrated by MCIF along with
	the copying of data into DRAM of the output generated. 

	Once all the layers are computed for convolution we will arrive at an upsampling stage which is to be handled by the host processor during this period the NVDLA would stall and wait for the upsampled data to be loaded back into 
	DRAM at which point further computation with this new data can proceed as it we were starting a new computation. 


	Now for some big questions and final summary ---- 
	1) What should be the size of SRAM?
	2) How many SRAM and for what purpose ? 
	3) How many stages of SRAM ?
	4) 














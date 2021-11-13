#include<iostream>
#include<tensorflow/c/c_api.h>

#include <fcntl.h>

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
//#include <unistd.h>

//#include<process.h>
#include<io.h>
struct model_t
{
	TF_Graph* graph;
	//TF_Buffer* buffer;
	TF_Session* session;
	TF_Status* status;

	TF_Output input, target, output;

	TF_Operation* init_operation, * train_operation, * save_operation, * restore_operation;
	TF_Output checkpoint_file;
};

int			modelCreate(model_t* model, const char* graf_def_file);
void		modelDestroy(model_t* model);
int			modelInit(model_t* model);
int			modelPedict(model_t* model, float* bach, int bach_size);
int			modelRunTrainStep(model_t* model);
void		nextBatchTraining(TF_Tensor** inputs_targets, TF_Tensor** targets_tensor);
enum		SaveOrRestore{ SAVE, RESTORE };
int			modelCheckpoint(model_t* model, const char* checkpoint_prefix, int type);

int			okay(TF_Status* status);
TF_Buffer*	readFile(const char* filename);
TF_Tensor*	scalarStringTensor(const char* data, TF_Status* status);
int			directoryExists(const char* dirname);

int main()
{
	
	return 0;
}

int modelCreate(model_t* model, const char* graph_def_file)
{
	model->status = TF_NewStatus();
	model->graph = TF_NewGraph();


	TF_SessionOptions* sessionOptions = TF_NewSessionOptions();
	model->session = TF_NewSession(model->graph, sessionOptions, model->status);
	TF_DeleteSessionOptions(sessionOptions);
	if(!okay(model->status))
	return 0;

	TF_Graph* g = model->graph;

	TF_Buffer* buffer = readFile(graph_def_file);
	if (buffer == nullptr)
		return 0;

	std::cout << "Read GraphDef of bytes:\t" << buffer->length << std::endl;
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(g, buffer, opts, model->status);
	TF_DeleteImportGraphDefOptions(opts);
	TF_DeleteBuffer(buffer);
	if (!okay(model->status)) return 0;

	model->input.oper = TF_GraphOperationByName(g, "input");
	model->input.index = 0;
	model->target.oper = TF_GraphOperationByName(g, "target");
	model->target.index = 0;
	model->output.oper = TF_GraphOperationByName(g, "output");
	model->output.index = 0;

	model->init_operation = TF_GraphOperationByName(g, "init");
	model->train_operation = TF_GraphOperationByName(g, "train");
	model->save_operation = TF_GraphOperationByName(g, "save/control_dependency");
	model->restore_operation = TF_GraphOperationByName(g, "save/restore_all");

	model->checkpoint_file.oper = TF_GraphOperationByName(g, "save/Const");
	model->checkpoint_file.index = 0;
	return 1;
}

void modelDestroy(model_t* model)
{
	TF_DeleteSession(model->session, model->status);
	okay(model->status);
	TF_DeleteGraph(model->graph);
	TF_DeleteStatus(model->status);
}

int modelInit(model_t* model)
{
	const TF_Operation* init_operation = { model->init_operation };
	TF_SessionRun(model->session, NULL,
		/* No inputs */
		NULL, NULL, 0,
		/* No outputs */
		NULL, NULL, 0,
		/* Just the init operation */
		&init_operation, 1,
		/* No metadata */
		NULL, model->status);

	return okay(model->status);
}

int modelPedict(model_t* model, float* batch, int batch_size)
{
	const int64_t dims[3] = { batch_size, 1, 1 };
	const size_t nbytes = batch_size * sizeof(float);
	TF_Tensor* t = TF_AllocateTensor(TF_FLOAT, dims, 3, nbytes);
	memcpy(TF_TensorData(t), batch, nbytes);
	memcpy(TF_TensorData(t), batch, nbytes);

	TF_Output inputs[1] = { model->input };
	TF_Tensor* input_values[1] = { t };
	TF_Output outputs[1] = { model->output };
	TF_Tensor* output_values[1] = { NULL };

	TF_SessionRun(model->session, NULL, inputs, input_values, 1, outputs,
		output_values, 1,
		/* No target operations to run */
		NULL, 0, NULL, model->status);
	TF_DeleteTensor(t);
	if (!okay(model->status)) return 0;

	if (TF_TensorByteSize(output_values[0]) != nbytes) {
		fprintf(stderr,
			"ERROR: Expected predictions tensor to have %zu bytes, has %zu\n",
			nbytes, TF_TensorByteSize(output_values[0]));
		TF_DeleteTensor(output_values[0]);
		return 0;
	}
	float* predictions = (float*)malloc(nbytes);
	memcpy(predictions, TF_TensorData(output_values[0]), nbytes);
	TF_DeleteTensor(output_values[0]);

	printf("Predictions:\n");
	for (int i = 0; i < batch_size; ++i) {
		std::cout << " x =\t" << batch[i] << "predicted y=\t" << predictions[i] << std::endl;
	}
	free(predictions);
	return 1;

}


void nextBatchTraining(TF_Tensor** inputs_tensor, TF_Tensor** targets_tensor)
{
#define BATCH_SIZE 10
	float inputs[BATCH_SIZE] = { 0 };
	float targets[BATCH_SIZE] = { 0 };

	for (int i = 0; i < BATCH_SIZE; i++)
	{
		inputs[i] = (float)rand() / (float)RAND_MAX;
		targets[i] = 3.0 * inputs[i] + 2.0;
	}
	const int64_t dims[] = { BATCH_SIZE, 1, 1 };
	size_t nbytes = BATCH_SIZE * sizeof(float);
	*inputs_tensor = TF_AllocateTensor(TF_FLOAT, dims, 3, nbytes);
	*targets_tensor = TF_AllocateTensor(TF_FLOAT, dims, 3, nbytes);
	memcpy(TF_TensorData(*inputs_tensor), inputs, nbytes);
	memcpy(TF_TensorData(*targets_tensor), targets, nbytes);
#undef BATCH_SIZE
}



int modelRunTrainStep(model_t* model)
{
	TF_Tensor* x, * y;
	nextBatchTraining(&x, &y);
	TF_Output inputs[] = { model->input, model->target };
	TF_Tensor* input_values[] = { x,y };
	const TF_Operation* train_op[] = { model->train_operation };
	TF_SessionRun(model->session, NULL, inputs, input_values, 2,
		/* No outputs */
		NULL, NULL, 0, train_op, 1, NULL, model->status);
	TF_DeleteTensor(x);
	TF_DeleteTensor(y);
	return okay(model->status);
}
int modelCheckpoint(model_t* model, const char* checkpoint_prefix, int type)
{
	return 0;
}

int okay(TF_Status* status)
{
	if (TF_GetCode(status) != TF_OK) {
		std::cout << stderr << "ERROR: %s\n" << TF_Message(status) << std::endl;
		return 0;
	}
	return 1;
}

TF_Buffer* readFile(const char* filename)
{
	int fd = _open(filename, 0);
	if (fd < 0) {
		perror("failed to open file: ");
		return NULL;
	}

	struct stat stat;
	if (fstat(fd, &stat) != 0) {
		perror("failed to read file: ");
		return NULL;
	}
	char* data = (char*)malloc(stat.st_size);
	size_t nread = read(fd, data, stat.st_size);
	if (nread < 0) {
		perror("failed to read file: ");
		free(data);
		return NULL;
	}
	if (nread != stat.st_size) {
		fprintf(stderr, "read %zd bytes, expected to read %zd\n", nread,
			stat.st_size);
		free(data);
		return NULL;
	}
	TF_Buffer* ret = TF_NewBufferFromString(data, stat.st_size);
	free(data);
	return ret;
}

TF_Tensor* scalarStringTensor(const char* data, TF_Status* status)
{
	size_t nbytes = 8 + TF_StringEncodedSize(strlen(str));
	return nullptr;
}

int directoryExists(const char* dirname)
{
	return 0;
}

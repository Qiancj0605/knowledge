bool read_TRT_File(const std::string& engineFile, IHostMemory*& trtModelStream)
{
     std::fstream file;
     std::cout << "loading filename from:" << engineFile << std::endl;
     nvinfer1::IRuntime* trtRuntime;
     //nvonnxparser::IPluginFactory* onnxPlugin = createPluginFactory(gLogger.getTRTLogger());
     file.open(engineFile, std::ios::binary | std::ios::in);
     file.seekg(0, std::ios::end);
     int length = file.tellg();
     std::cout << "length:" << length << std::endl;
     file.seekg(0, std::ios::beg);
     std::unique_ptr<char[]> data(new char[length]);
     file.read(data.get(), length);
     file.close();
     std::cout << "load engine done" << std::endl;
     std::cout << "deserializing" << std::endl;
     trtRuntime = createInferRuntime(gLogger.getTRTLogger());
     //ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length, onnxPlugin);
     ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length, nullptr);
     std::cout << "deserialize done" << std::endl;
     assert(engine != nullptr);
     std::cout << "The engine in TensorRT.cpp is not nullptr" <<std::endl;
     trtModelStream = engine->serialize();
     return true;

		 /*
		 // save engine file
		 nvinfer1::IHostMemory* data = engine->serialize();
		 std::ofstream file;
		 file.open(filename, std::ios::binary | std::ios::out);
		 cout << "writing engine file..." << endl;
		 file.write((const char*)data->data(), data->size());
		 cout << "save engine file done" << endl;
		 file.close();
		 */
 }


/*
 * 중앙대학교 대학원, 전자상거래 및 인터넷응용 연구실, 김준호
 * https://www.facebook.com/taki0112
 * taki0112@ec.cse.cau.ac.kr
 */

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


// 인수 -train a -output b

@SuppressWarnings("deprecation")
public class Word2Vec {
	private static final Logger log = LoggerFactory.getLogger(Word2Vec.class);
	private static final String input = "input_text.txt";
	/*
	 * 인풋문서는 전처리가 모두 완료되어야 합니다. (한줄에는 띄어쓰기로 구분되어 단어들로 구성)
	 * 한줄당 문서 1개를 의미
	 * Example ( Doc1 : 아이폰은 애플에서 만들었다. // Doc2 : 갤럭시는 삼성에서 만들었다.)
	 * Input Text
	 		아이폰 애플 만듬 or 아이폰 애플
	 		갤럭시 삼성 만듬 or 갤럭시 삼성
	 * 
	 */
	
	
	private static final String output = "Word_vec.txt";
	
	class VocabWord implements Comparable<VocabWord> {
		VocabWord(String word) {
			this.word = word;
		}
		int cn = 0;
		int codelen;
		int[] point = new int[MAX_CODE_LENGTH];
		long[] code = new long[MAX_CODE_LENGTH];
		String word;
		
		@Override
		public int compareTo(VocabWord that) {
			if(that==null) {
				return 1;
			}
			
			return that.cn - this.cn;
		}
		@Override
		public String toString() {
			return this.cn + ": " + this.word;
		}
	}
	
	private static final int MAX_STRING  = 100;
	private static final int EXP_TABLE_SIZE= 1000;
	private static final int MAX_EXP= 6;
	private static final int MAX_SENTENCE_LENGTH= 1000;
	private static final int MAX_CODE_LENGTH= 40;
	private static final int TABLE_SIZE = 100000000;
	
	// Maximum 30 * 0.7 = 21M words in the vocabulary
	private static final int VOCAB_HASH_SIZE = 30000000;
	
	private final int layerOneSize;
	private final File trainFile; 
	private final File outputFile;
	private final File saveVocabFile; 
	private final File readVocabFile;
	private final int window;
	private final int negative;
	private final int minCount; 
	private final int numThreads;
	private final int classes;
	private final boolean binary;
	private final boolean cbow;
	private final boolean noHs;
	private final float startingAlpha;
	private final float sample;
	private final float[] expTable;
	
	private int minReduce = 1;
	private int vocabMaxSize = 1000; 
	private VocabWord[] vocabWords = new VocabWord[vocabMaxSize];
	private int[] vocabHash = new int[VOCAB_HASH_SIZE];
	private Byte ungetc = null;
	
	private int vocabSize = 0; 	
	private long trainWords = 0;
	private long wordCountActual = 0;
	private int[] table;
	
	private float alpha;
	
	private float[] syn0; 
	private float[] syn1; 
	private float[] syn1neg; 
	
	private long start;
	
	public Word2Vec(Builder b) {
		this.trainFile = b.trainFile; 
		this.outputFile = b.outputFile; 
		this.saveVocabFile = b.saveVocabFile; 
		this.readVocabFile = b.readVocabFile; 
		this.binary = b.binary;
		this.cbow = b.cbow;
		this.noHs = b.noHs;
		this.startingAlpha = b.startingAlpha;
		this.sample = b.sample;		
		this.window = b.window;
		this.negative = b.negative;
		this.minCount = b.minCount; 
		this.numThreads = b.numThreads;
		this.classes = b.classes;
		this.layerOneSize = b.layerOneSize;
		
		float[] tempExpTable = new float[EXP_TABLE_SIZE];
		for (int i = 0; i < tempExpTable.length; i++) {
			// Precompute the exp() table
			tempExpTable[i] = (float) Math.exp((i / (float) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
			// Precompute f(x) = x / (x + 1)
			tempExpTable[i] = tempExpTable[i] / (tempExpTable[i] + 1);
		}
		expTable = tempExpTable;
	}
	private void readVocab() throws IOException {
		vocabSize = 0;
		try(DataInputStream is = new DataInputStream(new FileInputStream(readVocabFile))) {
			String word;
			while((word = readWord(is)) != null) {
				int a = addWordToVocab(word);
				vocabWords[a].cn = is.readInt();
				is.readChar();
			}
			sortVocab();			
			log.debug("Vocab size: {}", vocabSize);
			log.debug("Words in train file: {}", trainWords);			
		} catch(IOException ioe) {
			throw ioe;
		}
	}

	private void learnVocabFromTrainFile() throws IOException {
		for (int a = 0; a < VOCAB_HASH_SIZE; a++) {
			vocabHash[a] = -1;
		}
		vocabSize = 0;
		addWordToVocab("</s>");
		try (DataInputStream is = new DataInputStream(new FileInputStream(trainFile))) {
			while (true) {
				String word = readWord(is);
				if (word==null) {
					break;
				}
				trainWords++;
				if(log.isTraceEnabled() && trainWords % 100000 == 0) {
					log.trace("{}K training words processed.", (trainWords/1000));
				}
				int i = searchVocab(word);
				if (i == -1) {
					i = addWordToVocab(word);
					vocabWords[i].cn = 1;
				} else
					vocabWords[i].cn++;
				if (vocabSize > VOCAB_HASH_SIZE * 0.7) {
					reduceVocab();
				}
			}
		} catch (IOException ioe) {
			throw ioe;
		}
		sortVocab();
		log.debug("Vocab size: {}", vocabSize);
		log.debug("Words in train file: {}", trainWords);
	}

	private void saveVocab() throws IOException {
		saveVocabFile.delete();
		try (FileWriter fw = new FileWriter(saveVocabFile)) {
			//Don't output the </s>, at element zero.
			for (int i = 1; i < vocabSize; i++) {
				fw.write(vocabWords[i].word);
				fw.write(" ");
				fw.write("" + vocabWords[i].cn);
				fw.write("\n");
			}
		}
	}
	private void initNet() {
		syn0 = new float[vocabSize * layerOneSize];
		if(!noHs) {
			syn1 = new float[vocabSize * layerOneSize];
			for(int b=0 ; b < layerOneSize ; b++) {
				for(int a=0 ; a<vocabSize ; a++) {
					syn1[a * layerOneSize + b] = 0;
				}
			}
		}
		if(negative>0) {
			syn1neg = new float[vocabSize * layerOneSize];
			for(int b=0 ; b < layerOneSize ; b++) {
				for(int a=0 ; a<vocabSize ; a++) {
					syn1neg[a * layerOneSize + b] = 0;
				}
			}
		}
		for(int b=0 ; b< layerOneSize ; b++) {
			for(int a=0 ; a<vocabSize ; a++) {
				syn0[a * layerOneSize + b] = (float) (Math.random() - 0.5) / layerOneSize;
			}
		}
		createBinaryTree();
	}
	
	// Create binary Huffman tree using the word counts
	// Frequent words will have short uniqe binary codes
	private void createBinaryTree() {
		//TODO: vocabSize.length cannot be longer than 1.2b .  Maybe use 2 arrays to allow this to be 2.4b?
		long[] count = new long[vocabSize * 2 + 1];
		long[] binary = new long[vocabSize * 2 + 1];
		int[] parentNode = new int[vocabSize * 2 + 1];
		for(int a=0 ; a<vocabSize ; a++) {
			count[a] = vocabWords[a].cn;
		}
		for(int a=vocabSize ; a<vocabSize*2 ; a++) {
			count[a] = 1_000_000_000_000_000L; //1e15
		}
		int pos1 = vocabSize - 1;
		int pos2 = vocabSize;
		int min1i;
		int min2i;
		
		// Following algorithm constructs the Huffman tree by adding one node at a time
		for (int a = 0; a < vocabSize - 1; a++) {
			// First, find two smallest nodes 'min1, min2'
			if (pos1 >= 0) {
				if (count[pos1] < count[pos2]) {
					min1i = pos1;
					pos1--;
				} else {
					min1i = pos2;
					pos2++;
				}
			} else {
				min1i = pos2;
				pos2++;
			}
			if (pos1 >= 0) {
				if (count[pos1] < count[pos2]) {
					min2i = pos1;
					pos1--;
				} else {
					min2i = pos2;
					pos2++;
				}
			} else {
				min2i = pos2;
				pos2++;
			}
			count[vocabSize + a] = count[min1i] + count[min2i];
			parentNode[min1i] = vocabSize + a;
			parentNode[min2i] = vocabSize + a;
			binary[min2i] = 1;
		}
		
		// Now assign binary code to each vocabulary word
		long[] code = new long[MAX_CODE_LENGTH];
		int[] point = new int[MAX_CODE_LENGTH];
		for (int a = 0; a < vocabSize; a++) {
			int b = a;
			int i = 0;
			while (true) {
				code[i] = binary[b];
				point[i] = b;
				i++;
				b = parentNode[b];
				if (b == vocabSize * 2 - 2)
					break;
			}
			vocabWords[a].codelen = i;
			vocabWords[a].point[0] = vocabSize - 2;
			for (b = 0; b < i; b++) {
				vocabWords[a].code[i - b - 1] = code[b];
				vocabWords[a].point[i - b] = point[b] - vocabSize;
			}
		}		
	}
	private void initUnigramTable() {
		long trainWordsPow = 0;
		float power = 0.75F;
		for (int a = 0; a < vocabSize; a++) { 
			trainWordsPow += Math.pow(vocabWords[a].cn, power);
		}
		int i = 0;
		float d1 = (float) Math.pow(vocabWords[i].cn, power) / (float) trainWordsPow;
		for (int a = 0; a < TABLE_SIZE; a++) {
			table[a] = i;
			if (a / (float) TABLE_SIZE > d1) {
				i++;
				d1 += Math.pow(vocabWords[i].cn, power) / (float) trainWordsPow;
			}
			if (i >= vocabSize) {
				i = vocabSize - 1;
			}
		}
	}
	
	//DataOutputStream#writeFloat writes the high byte first
	//but let's write the low byte first to give ourselves a better chance of
	//compatibility with the original c++ code
	private void writeFloat(float f, DataOutputStream out) throws IOException {
		int v = Float.floatToIntBits(f);
		out.write((v >>>  0) & 0xFF);
		out.write((v >>>  8) & 0xFF);
		out.write((v >>> 16) & 0xFF);
		out.write((v >>> 24) & 0xFF);
	}

	public void trainModel() {
		if(trainFile==null && readVocabFile==null) {
			throw new IllegalStateException("You must supply either a trainFile or a readVocabFile.");
		}
		alpha = startingAlpha;
		if(readVocabFile!=null) {
			try {
				log.info("Reading vocabulary from file {}.", readVocabFile);
				readVocab();
			} catch(IOException ioe) {
				log.error("There was a problem reading the vocabulary file.", ioe);
				return;
			}
		} else {
			log.info("Starting training using file {}.", trainFile);
			try {
				learnVocabFromTrainFile();
			} catch(IOException ioe) {
				log.error("There was a problem reading the training file.", ioe);
				return;
			}
		}
		if(saveVocabFile!=null) {
			try {
				saveVocab();
			} catch(IOException ioe) {
				log.error("There was a problem writing the vocabulary file.", ioe);
				return;
			}
		}
		if(outputFile==null) {
			return;
		}
		initNet();
		if(negative>0) {
			initUnigramTable();
		}
		start = System.nanoTime();
		//TODO: theads
		try {
			trainModelThread(0);
		} catch(IOException ioe) {
			log.error("There was a problem reading the training file.", ioe);
			return;
		}
		outputFile.delete();
		NumberFormat vectorTextFormat = new DecimalFormat("#.######");
		try(DataOutputStream os = new DataOutputStream(new FileOutputStream(outputFile))) {
			if(classes==0) {
				// Save the word vectors
				os.writeBytes("" + vocabSize + " " + layerOneSize + "\n");
				for (int a = 0; a < vocabSize; a++) {
					os.writeBytes(vocabWords[a].word);
					os.writeBytes(" ");
					if (binary) {
						for (int b = 0; b < layerOneSize; b++) {
							writeFloat(syn0[a * layerOneSize + b], os);
						}
					} else {
						for (int b = 0; b < layerOneSize; b++) {
							int index = a * layerOneSize + b;
							float value = syn0[index];
							os.writeBytes(vectorTextFormat.format(value) + " ");
						}
					}
					os.writeBytes("\n");

				}
				os.writeBytes("\n");
			} else {
				// Run K-means on the word vectors
				if(classes*layerOneSize > Integer.MAX_VALUE) {
					throw new RuntimeException("Number of classes times the size of Layer One cannot be greater than " + Integer.MAX_VALUE + " (" + classes + " * " + layerOneSize + ")");
				}
				int[] cl = new int[vocabSize];
				float[] cent = new float[classes * layerOneSize];
				int[] centcn = new int[classes];
				int numIterations = 10;				
				
				for (int a = 0; a < vocabSize; a++) { 
					cl[a] = a % classes;
				}
				for(int a = 0; a<numIterations ; a++) {
					for(int b=0 ; b<(classes * layerOneSize) ; b++) {
						cent[b] = 0;
					}
					for(int b=0 ; b<classes ; b++) {
						centcn[b] = 1;
					}
					for(int c=0 ; c<vocabSize ; c++) {
						for(int d=0 ; d<layerOneSize ; d++) {
							cent[layerOneSize * cl[c] + d] += syn0[c * layerOneSize + d];
						}
						centcn[cl[c]]++;
					}
					for(int b=0 ; b < classes ; b++) {
						float closev = 0;
						for(int c=0 ; c<layerOneSize ; c++) {
							cent[layerOneSize * b + c] /= centcn[b];
							closev += cent[layerOneSize * b + c] * cent[layerOneSize * b + c];  //TODO: ^2 ??
						}
						closev = (float) Math.sqrt(closev);
						for(int c=0 ; c < layerOneSize ; c++) {
							cent[layerOneSize * b + c] /= closev;
						}
					}
					for(int c=0 ; c<vocabSize ; c++) {
						float closev = -10;
						int closeid = 0;
						for(int d=0 ; d < classes ; d++) {
							float x = 0;
							for(int b=0 ; b<layerOneSize ; b++) {
								x+= cent[layerOneSize * d * b] * syn0[c * layerOneSize + b];
							}
							if (x > closev) {
								closev = x;
								closeid = d;
							}
						}
						cl[c] = closeid;
					}					
				}
				// Save the K-means classes
				for(int a=0 ; a< vocabSize ; a++) {
					os.writeBytes(vocabWords[a].word);
					os.writeBytes(" ");
					os.writeInt(cl[a]);
				}
			}
		} catch(IOException ioe) {
			log.error("There was a problem writing the output file", ioe);
			return;
		}		
	}
	private void trainModelThread(int id) throws IOException {
		try(RandomAccessFile raf = new RandomAccessFile(trainFile, "rw")) {
			if(id>0) {
				raf.seek(raf.length() / (numThreads * id));
			}
			long wordCount = 0;
			long lastWordCount = 0;
			int word = 0;
			int target = 0;
			int label = 0;
			int sentenceLength = 0;
			int sentencePosition = 0;
			int nextRandom = id;
			int[] sen = new int[MAX_SENTENCE_LENGTH + 1];
			float[] neu1 = new float[layerOneSize];
			float[] neu1e = new float[layerOneSize];
			
			NumberFormat alphaFormat = new DecimalFormat("0.000000");
			NumberFormat logPercentFormat = new DecimalFormat("#0.00%");
			NumberFormat wordsPerSecondFormat = new DecimalFormat("00.00k");
			long now = System.nanoTime();
			while(true) {
				if (wordCount - lastWordCount > 10000) {
					wordCountActual += wordCount - lastWordCount;
					lastWordCount = wordCount;
					if (log.isTraceEnabled()) {
						now = System.nanoTime();
						log.trace("Alpha: {}", alphaFormat.format(alpha));
						log.trace("Progress: {} ", logPercentFormat.format((float) wordCountActual / (trainWords + 1)));
						log.trace(
								"Words/thread/sec: {}\n",
								wordsPerSecondFormat.format((float) wordCountActual / (float) (now - start + 1)
										* 1000000));
					}
					alpha = startingAlpha * (1 - wordCountActual / (float) (trainWords + 1));
					if (alpha < startingAlpha * 0.0001F) {
						alpha = startingAlpha * 0.0001F;
					}
				}
				if(sentenceLength==0) {
					while(true) {
						word = readWordIndex(raf);
						if(word==-1) {
							break;
						}
						wordCount++;
						if(word==0) {
							break;
						}
						// The subsampling randomly discards frequent words while keeping the ranking same
				        if (sample > 0) {
				        	float ran = (float) (Math.sqrt(vocabWords[word].cn / (sample * trainWords)) + 1) * (sample * trainWords) / vocabWords[word].cn;
				        	nextRandom = (int) (nextRandom * 25214903917L + 11);
				            if (ran < ((nextRandom & 0xFFFF) / (float) 65536)) { 
				            	continue;
				            }
				        }
				        sen[sentenceLength] = word;
				        sentenceLength++;
				        if (sentenceLength >= MAX_SENTENCE_LENGTH) {
				        	break;
				        }
					}
					sentencePosition = 0;
				}
				if(raf.getFilePointer()==raf.length()) {
					break;
				}
				if(wordCount > trainWords / numThreads) {
					break;
				}
				word = sen[sentencePosition];
				for(int c=0 ; c<layerOneSize ; c++) {
					neu1[c] = 0;
					neu1e[c] = 0;
				}
				nextRandom = (int) (nextRandom * 25214903917L + 11);
			    int b = nextRandom % window;
			    if(cbow) {
			    	// in -> hidden
			    	for(int a = b ; b < window * 2 + 1 - b ; a++) {
			    		if(a != window) {
			    			int c = sentencePosition - window + a;
			    	        if (c < 0) {
			    	        	continue;
			    	        }
			    	        if (c >= sentenceLength) {
			    	        	continue;
			    	        }
			    	        int lastWord = sen[c];
			    	        for (c = 0; c < layerOneSize; c++)  {
			    	        	neu1[c] += syn0[c + lastWord * layerOneSize];
			    	        }
			    		}
			    	}
			    	if (!noHs) {
			    		for(int d=0 ; d< vocabWords[word].codelen ; d++) {
			    			float f=0;
			    			int l2 = vocabWords[word].point[d] * layerOneSize;
			    			// Propagate hidden -> output
			    			for (int c = 0; c < layerOneSize; c++) {
			    				f += neu1[c] * syn1[c + l2];
			    			}
			    	        if (f <= -1 * MAX_EXP || f >= MAX_EXP) {
			    	        	continue;
			    	        }
			    	        f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
			    	        // 'g' is the gradient multiplied by the learning rate
			    	        float g = (1 - vocabWords[word].code[d] - f) * alpha;
			    	        // Propagate errors output -> hidden
			    	        for (int c = 0; c < layerOneSize; c++) {
			    	        	neu1e[c] += g * syn1[c + l2];
			    	        }
			    	        // Learn weights hidden -> output 
			    	        for (int c = 0; c < layerOneSize; c++) { 
			    	        	syn1[c + l2] += g * neu1[c];
			    	        }
			    		}			    		
			    	}
			    	// NEGATIVE SAMPLING
			        if (negative > 0) {
			        	for (int d = 0; d < negative + 1; d++) {			        
				          if (d == 0) {
				            target = word;
				            label = 1;
				          } else {
				            nextRandom = (int) (nextRandom * 25214903917L + 11);
				            target = table[(nextRandom >> 16) % TABLE_SIZE];
				            if (target == 0) { 
				            	target = nextRandom % (vocabSize - 1) + 1;
				            }
				            if (target == word) {
				            	continue;
				            }
				            label = 0;
				          }
				          int l2 = target * layerOneSize;
				          int f = 0;
				          for (int c = 0; c < layerOneSize; c++) {
				        	  f += neu1[c] * syn1neg[c + l2];
				          }
				          float g;
				          if (f > MAX_EXP) {
				        	  g = (label - 1) * alpha;
				          } else if (f < -MAX_EXP) {
				        	  g = (label - 0) * alpha;
				          } else {
				        	  g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
				          }
				          for (int c = 0; c < layerOneSize; c++) {
				        	  neu1e[c] += g * syn1neg[c + l2];
				          }
				          for (int c = 0; c < layerOneSize; c++) {
				        	  syn1neg[c + l2] += g * neu1[c];
				          }
			        	}
			        }
			        // hidden -> in
			        for (int a = b; a < window * 2 + 1 - b; a++) {
			        	if (a != window) {			        
				          int c = sentencePosition - window + a;
				          if (c < 0 || c >= sentenceLength) {
				        	  continue;
				          }
				          int lastWord = sen[c];
				          for (c = 0; c < layerOneSize; c++) {
				        	  syn0[c + lastWord * layerOneSize] += neu1e[c];
				          }
			        	}
			        }
			    } else { //train skip-gram
			    	for (int a = b; a < window * 2 + 1 - b; a++) {
			    		if (a != window) {		    		
				            int lastWordIndex = sentencePosition - window + a;
				            if (lastWordIndex < 0 || lastWordIndex >= sentenceLength) {
				            	continue;
				            }
				            int lastWord = sen[lastWordIndex];
				            int l1 = lastWord * layerOneSize;
				            for (int c = 0; c < layerOneSize; c++) {
				            	neu1e[c] = 0;
				            }
				            // HIERARCHICAL SOFTMAX
				            if (!noHs) {
				            	for (int d = 0; d < vocabWords[word].codelen; d++) {
					              float f = 0;
					              int l2 = vocabWords[word].point[d] * layerOneSize;
					              // Propagate hidden -> output
					              for (int c = 0; c < layerOneSize; c++) {
					            	  f += syn0[c + l1] * syn1[c + l2];
					              }
					              if (f <= -MAX_EXP || f >= MAX_EXP) { 
					            	  continue;					              
					              }
					              f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					              // 'g' is the gradient multiplied by the learning rate
					              float g = (1 - vocabWords[word].code[d] - f) * alpha;
					              // Propagate errors output -> hidden
					              for (int c = 0; c < layerOneSize; c++) { 
					            	  neu1e[c] += g * syn1[c + l2];
					              }
					              // Learn weights hidden -> output
					              for (int c = 0; c < layerOneSize; c++) {
					            	  syn1[c + l2] += g * syn0[c + l1];
					              }
				            	}
				            }
				            // NEGATIVE SAMPLING
				            if (negative > 0) {
				            	for (int d = 0; d < negative + 1; d++) {				            
					              if (d == 0) {
					                target = word;
					                label = 1;
					              } else {
					            	  nextRandom = (int) (nextRandom * 25214903917L + 11);
							          target = table[(nextRandom >> 16) % TABLE_SIZE];
							          if (target == 0) { 
							        	  target = nextRandom % (vocabSize - 1) + 1;
							          }
							          if (target == word) {
							        	  continue;
							          }
							          label = 0;
					              }
					              int l2 = target * layerOneSize;
					              int f = 0;
					              for (int c = 0; c < layerOneSize; c++) {
					            	  f += syn0[c + l1] * syn1neg[c + l2];
					              }
					              float g;
					              if (f > MAX_EXP) {
					            	  g = (label - 1) * alpha;
					              } else if (f < -MAX_EXP)  {
					            	  g = (label - 0) * alpha;
					              } else {
					            	  g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
					              }
					              for (int c = 0; c < layerOneSize; c++) {
					            	  neu1e[c] += g * syn1neg[c + l2];
					              }
					              for (int c = 0; c < layerOneSize; c++) {
					            	  syn1neg[c + l2] += g * syn0[c + l1];
					              }
				            	}
				            }
				            // Learn weights input -> hidden
				            for (int c = 0; c < layerOneSize; c++) {
				            	syn0[c + l1] += neu1e[c];
				            }
				        }
				    }
			    }
			    sentencePosition++;
			    if (sentencePosition >= sentenceLength) {
			        sentenceLength = 0;
			        continue;
			    }			    
			}
		} catch(IOException ioe) {
			throw ioe;
		}
	}
	
	// Reduces the vocabulary by removing infrequent tokens
	private void reduceVocab() {
		int b=0;
		for (int a = 0; a < vocabSize; a++) {
			if (vocabWords[a].cn > minReduce) {		
				vocabWords[b].cn = vocabWords[a].cn;
				vocabWords[b].word = vocabWords[a].word;
				b++;
			}
		} 
		vocabSize = b;
		for (int a = 0; a < VOCAB_HASH_SIZE; a++) {
			vocabHash[a] = -1;
		}
		for (int a = 0; a < vocabSize; a++) {
			// Hash will be re-computed, as it is not actual
			int hash = getWordHash(vocabWords[a].word);
			while (vocabHash[hash] != -1) {
				hash = (hash + 1) % VOCAB_HASH_SIZE;
				hash = Math.abs(hash);
			}			
			vocabHash[hash] = a;
		}
		minReduce++;
	}
	// Sorts the vocabulary by frequency using word counts
	private void sortVocab() {
		// Sort the vocabulary and keep </s> at the first position
		Arrays.sort(vocabWords, 1, vocabSize - 1);
		for (int a = 0; a < vocabHash.length; a++) {
			vocabHash[a] = -1;
		}

		trainWords = 0;
		int originalVocabSize = vocabSize;
		List<VocabWord> wordList = new ArrayList<VocabWord>(originalVocabSize);
		int aa=0;
		for (int a = 0; a < originalVocabSize; a++) {
			VocabWord vw = vocabWords[a];
			// Words occurring less than min_count times will be discarded from the vocab
			if (vw.cn < minCount && vw.cn > 0) {
				vocabSize--;
			} else {
				// Hash will be re-computed, as after the sorting it is not actual
				int hash = getWordHash(vw.word);
				while (vocabHash[hash] != -1) {
					hash = (hash + 1) % VOCAB_HASH_SIZE;
					hash = Math.abs(hash);
				}
				vocabHash[hash] = aa;
				trainWords += vw.cn;
				wordList.add(vw);
				aa++;
			}
		}
		vocabWords = wordList.toArray(new VocabWord[wordList.size()]);
	}
	
	private int addWordToVocab(String word) {
		int length = word.length() + 1;
		if(length > MAX_STRING) {
			length = MAX_STRING;
		}
		vocabWords[vocabSize] = new VocabWord(word);
		vocabSize++;
		
		// Reallocate memory if needed
		if (vocabSize + 2 >= vocabMaxSize) {
			vocabMaxSize += 1000;
			VocabWord[] vocabWords1 = new VocabWord[vocabMaxSize];
			System.arraycopy(vocabWords, 0, vocabWords1, 0, vocabWords.length);
			vocabWords = vocabWords1;
		}
		int hash = getWordHash(word);
		while (vocabHash[hash] != -1) {
			hash = (hash + 1) % VOCAB_HASH_SIZE;
			hash = Math.abs(hash);
		}
		vocabHash[hash] = vocabSize - 1;
		return vocabSize - 1;
	}
	private int getWordHash(String word) {
		int hash = 0;
		for (int a = 0; a < word.length(); a++) {
			hash = hash * 257 + word.charAt(a);
		}
		hash = hash % VOCAB_HASH_SIZE;
		return Math.abs(hash);
	}
	
	// Returns position of a word in the vocabulary; if the word is not found, returns -1
	private int searchVocab(String word) {
	  int hash = getWordHash(word);
	  while (true) {
	    if (vocabHash[hash] == -1) {
	    	return -1;
	    }
	    if(word.equals(vocabWords[vocabHash[hash]].word)) {
	    	return vocabHash[hash];
	    }
	    hash = (hash + 1) % VOCAB_HASH_SIZE;
	    hash = Math.abs(hash);
	  }
	}
	
	private int readWordIndex(RandomAccessFile raf) throws IOException {
		String word = readWord(raf);
		if(raf.length()==raf.getFilePointer()) {
			return -1;
		}
		return searchVocab(word);
	}
	
	private String readWord(DataInput dataInput) throws IOException {
		StringBuilder sb = new StringBuilder();
		while(true) {			
			byte ch;			
			if(ungetc != null) {
				ch = ungetc;
				ungetc = null;
			} else {
				try {
					ch = dataInput.readByte();
				} catch(EOFException eofe) {
					break;
				}
			}
			if(ch=='\r') {
				continue;
			}
			if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
				if (sb.length()>0 ) {	
					if(ch == '\n') {
						ungetc = ch;
					}
					break;													
				}
				if (ch == '\n') {
					return "</s>";
				} else {
					continue;
				}
			}
			sb.append((char) ch);
			
			// Truncate too long words
			if (sb.length() >= MAX_STRING - 1) {
				sb.deleteCharAt(sb.length()-1);
			}				
		}
		String word = sb.length()==0 ? null : sb.toString();
		return word;
	}
	// 변수설명
	public static class Builder {		
		private File trainFile = null; 
		private File outputFile = null; 
		private File saveVocabFile = null; 
		private File readVocabFile = null; 
		private boolean binary = false;
		private boolean cbow = false; // if false, use cbow model.. else, use skip-gram model
		private boolean noHs = false;
		private float startingAlpha = 0.025F; //0.025F, learning rate
		private float sample = 0.0F;		
		private int window = 5; // Surrounding word
		private int negative = 0; // If negative = 0, use Hierarchical Softmax ... 
		// default negative = 5 ~ 10
		private int minCount = 5; // mincount word, if you use all word then minCount = 0
		private int numThreads = 1;
		private int classes = 0;
		private int layerOneSize = 200; // vecter size
		public Builder trainFile(String trainFile) {
			this.trainFile = new File(trainFile);
			return this;
		}
		public Builder outputFile(String outputFile) {
			this.outputFile = new File(outputFile);
			return this;
		}
		public Builder saveVocabFile(String saveVocabFile) {
			this.saveVocabFile = new File(saveVocabFile);
			return this;
		}
		public Builder readVocabFile(String readVocabFile) {
			this.readVocabFile = new File(readVocabFile);
			return this;
		}
		public Builder binary() {
			this.binary = true;
			return this;
		}
		public Builder cbow() {
			this.cbow = true;
			return this;
		}
		public Builder noHs() {
			this.noHs = true;
			return this;
		}
		public Builder startingAlpha(float startingAlpha) {
			this.startingAlpha = startingAlpha;
			return this;
		}
		public Builder sample(float sample) {
			this.sample = sample;
			return this;
		}
		public Builder window(int window) {
			this.window = window;
			return this;
		}
		public Builder negative(int negative) {
			this.negative = negative;
			return this;
		}
		public Builder minCount(int minCount) {
			this.minCount = minCount;
			return this;
		}
		public Builder numThreads(int numThreads) {
			this.numThreads = numThreads;
			return this;
		}
		public Builder classes(int classes) {
			this.classes = classes;
			return this;
		}
		public Builder layerOneSize(int layerOneSize) {
			this.layerOneSize = layerOneSize;
			return this;
		}
	}
	
	@SuppressWarnings({ "static-access" })
	public static void Start(String[] args) {
		Builder b = new Builder();
		Options options = new Options();
		options.addOption(OptionBuilder.hasArg().withArgName("file")
				.withDescription("Use text data from <file> to train the model").create("train"));
		options.addOption(OptionBuilder.hasArg().withArgName("file")
				.withDescription("Use <file> to save the resulting word vectors / word clusters").create("output"));
		options.addOption(OptionBuilder.hasArg().withArgName("int")
				.withDescription("Set size of word vectors; default is " + b.layerOneSize).create("size"));
		options.addOption(OptionBuilder.hasArg().withArgName("int")
				.withDescription("Set max skip length between words; default is " + b.window).create("window"));
		options.addOption(OptionBuilder
				.hasArg()
				.withArgName("int")
				.withDescription(
						"Set threshold for occurrence of words (0=off). Those that appear with higher frequency in the training data will be randomly down-sampled; default is "
								+ b.sample + ", useful value is 1e-5").create("sample"));
		options.addOption(new Option("noHs", false, "Disable use of Hierarchical Softmax; " + (b.noHs ? "off" : "on")
				+ " by default"));
		options.addOption(OptionBuilder
				.hasArg()
				.withArgName("int")
				.withDescription(
						"Number of negative examples; default is " + b.negative
								+ ", common values are 5 - 10 (0 = not used)").create("negative"));
		options.addOption(OptionBuilder.hasArg().withArgName("int")
				.withDescription("Use <int> threads (default " + b.numThreads + ")").create("threads"));
		options.addOption(OptionBuilder.hasArg().withArgName("int")
				.withDescription("This will discard words that appear less than <int> times; default is " + b.minCount)
				.create("minCount"));
		options.addOption(OptionBuilder.hasArg().withArgName("float")
				.withDescription("Set the starting learning rate; default is " + b.startingAlpha).create("startingAlpha"));
		options.addOption(OptionBuilder
				.hasArg()
				.withArgName("int")
				.withDescription(
						"Number of word classes to output, or 0 to output word vectors; default is " + b.classes)
				.create("classes"));
		options.addOption(new Option("binary", false, "Save the resulting vectors in binary moded; "
				+ (b.binary ? "on" : "off") + " by default"));
		options.addOption(OptionBuilder.hasArg().withArgName("file")
				.withDescription("The vocabulary will be saved to <file>").create("saveVocab"));
		options.addOption(OptionBuilder.hasArg().withArgName("file")
				.withDescription("The vocabulary will be read from <file>, not constructed from the training data")
				.create("readVocab"));
		options.addOption(new Option("cbow", false, "Use the continuous bag of words model; " + (b.cbow ? "on" : "off")
				+ " by default (skip-gram model)"));

		CommandLineParser parser = new PosixParser();
		try {
			CommandLine cl = parser.parse(options, args);
			if (cl.getOptions().length == 0) {
				new HelpFormatter().printHelp(Word2Vec.class.getSimpleName(), options);
				System.exit(0);
			}
			if (cl.hasOption("size")) {
				b.layerOneSize = Integer.parseInt(cl.getOptionValue("size"));
			}
			if (cl.hasOption("train")) {
				//b.trainFile = new File(cl.getOptionValue("train"));
				b.trainFile = new File(input);
			}
			if (cl.hasOption("saveVocab")) {
				b.saveVocabFile = new File(cl.getOptionValue("saveVocab"));
			}
			if (cl.hasOption("readVocab")) {
				b.readVocabFile = new File(cl.getOptionValue("readVocab"));
			}
			if (cl.hasOption("binary")) {
				b.binary = true;
			}
			if (cl.hasOption("cbow")) {
				b.cbow = true;
			}
			if (cl.hasOption("startingAlpha")) {
				b.startingAlpha = Float.parseFloat(cl.getOptionValue("startingAlpha"));
			}
			if (cl.hasOption("output")) {
			//	b.outputFile = new File(cl.getOptionValue("output"));
				b.outputFile = new File(output);
			}
			if (cl.hasOption("window")) {
				b.window = Integer.parseInt(cl.getOptionValue("window"));
			}
			if (cl.hasOption("sample")) {
				b.sample = Float.parseFloat(cl.getOptionValue("sample"));
			}
			if (cl.hasOption("noHs")) {
				b.noHs = true;
			} 
			if (cl.hasOption("negative")) {
				b.negative = Integer.parseInt(cl.getOptionValue("negative"));
			}
			if (cl.hasOption("threads")) {
				b.numThreads = Integer.parseInt(cl.getOptionValue("threads"));
			}
			if (cl.hasOption("minCount")) {
				b.minCount = Integer.parseInt(cl.getOptionValue("minCount"));
			}
			if (cl.hasOption("classes")) {
				b.classes = Integer.parseInt(cl.getOptionValue("classes"));
			}
		} catch (Exception e) {
			System.err.println("Parsing command-line arguments failed. Reason: " + e.getMessage());
			new HelpFormatter().printHelp("word2vec", options);
			System.exit(1);
		}
		Word2Vec word2vec = new Word2Vec(b);
		word2vec.trainModel();
		//System.exit(0);
	}
}

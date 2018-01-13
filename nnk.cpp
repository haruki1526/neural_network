#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define run() ( (double)rand() / RAND_MAX / 1.0)
#define INPUT_LAYER 64 //input_layer number
#define HIDDEN_LAYER 64 //hedden_layer number
#define OUTPUT_LAYER 10 //output_layer number
#define HIDDEN_LAYER_NUM 5
#define N 0.3 //learninig_num
#define ALPHA 0.1
#define LAYER_SUM 7
#define LEARN_SUM 100000


double sigmoid(double x)//sigmoid def 
{
		double f;
		f=1.0/(1.0+exp(-x));
		return f;

}

double d_sigmoid(double u)
{
	double f;
	f=sigmoid(u) * (1 - sigmoid(u));

	return f;

}


int main(){
       	double teach_out[OUTPUT_LAYER]={1,0,0,0,0,0,0,0,0,0};

	double output_layer[OUTPUT_LAYER]={0,0,0,0,0,0,0,0,0,0};
	
	double teach_input[64]={0,0,0,1,1,0,0,0,
	    		      	0,0,1,0,0,1,0,0,
			      	0,0,1,0,0,1,0,0,
			      	0,0,1,0,0,1,0,0,
			      	0,0,1,0,0,1,0,0,
			      	0,0,1,0,0,1,0,0,
			      	0,0,1,0,0,1,0,0,
			      	0,0,0,1,1,0,0,0};
	
	double input_layer[64]={0,0,0,1,1,0,0,0,
	    		      	0,0,1,0,0,1,0,0,
			      	0,0,1,0,0,1,0,0,
			      	0,0,1,0,0,1,0,0,
			      	0,0,1,0,0,1,0,0,
			      	0,0,1,0,0,1,0,0,
			      	0,0,1,0,0,1,0,0,
			      	0,0,0,1,1,0,0,0};


	double weight[LAYER_SUM][HIDDEN_LAYER][HIDDEN_LAYER];
	double old_weight[LAYER_SUM][HIDDEN_LAYER][HIDDEN_LAYER];
	double h[LAYER_SUM][HIDDEN_LAYER];
	double del_h[LAYER_SUM][HIDDEN_LAYER];
	double old_del_h[LAYER_SUM][HIDDEN_LAYER];



	srand((unsigned)time(NULL));
	int i;
	int j;
	int n;
	int k;
	int l;
	double sum;
	
	
	
	double x[LAYER_SUM][HIDDEN_LAYER];
	double u[LAYER_SUM][HIDDEN_LAYER];
	double del_weight[LAYER_SUM][HIDDEN_LAYER][HIDDEN_LAYER];
	double old_del_weight[LAYER_SUM][HIDDEN_LAYER][HIDDEN_LAYER];	
	double teach_signal[LAYER_SUM][HIDDEN_LAYER];   

	//given random_num for weight
	for( n=0; n < LAYER_SUM; n++){

		for( i=0; i < HIDDEN_LAYER; i++){
			
			for( j=0; j < HIDDEN_LAYER; j++){
			
	
				weight[n][i][j] = run();
				old_del_weight[n][i][j] = 0;
			}
		}
	}

	for( n=0; n < LAYER_SUM; n++){

		for( i=0; i < HIDDEN_LAYER; i++ ){

			h[n][i] = 0.5;
			old_del_h[n][i] = 0;

		}
	}


	//learnig start
	for(l=0; l < LEARN_SUM; l++){
		printf("%d\n",l);			
		//output by input_layer
		for( i=0; i < INPUT_LAYER; i++){
			x[0][i] = teach_input[i];
		}
		
		//output by hidden_layer 	
		for( n=1; n < LAYER_SUM-1; n++){
			for( i=0; i < HIDDEN_LAYER; i++){
			       u[n][i] = 0;	
				for( j=0; j < HIDDEN_LAYER; j++){
					u[n][i] += weight[n][i][j] * x[n-1][j];
				}

				x[n][i] = sigmoid( u[n][i] - h[n][i] );

			}
		}
		
		//output by output_layer
		for( i=0; i < OUTPUT_LAYER; i++ ){
			u[LAYER_SUM-1][i] = 0;
			for( j=0; j < HIDDEN_LAYER; j++ ){
				u[LAYER_SUM-1][i] += weight[LAYER_SUM-1][i][j] * x[LAYER_SUM-1][j];

			}
			x[LAYER_SUM-1][i] = sigmoid( u[LAYER_SUM-1][i] - h[LAYER_SUM-1][i] );

		

		}
	
		//teach_signal by output_layer
		
		for( i=0; i < OUTPUT_LAYER; i++ ){
			teach_signal[LAYER_SUM-1][i] = ( teach_out[i] - x[LAYER_SUM-1][i] ) * d_sigmoid( u[LAYER_SUM-1][i]);	
		}
		
		//teach_signal by hidden_layer 
		
		for( i=0; i < HIDDEN_LAYER; i++ ){
			sum = 0;
			for( k=0; k < OUTPUT_LAYER; k++ ){
				sum += teach_signal[HIDDEN_LAYER_NUM+1][k] * weight[HIDDEN_LAYER_NUM+1][k][i];
			}
			
			teach_signal[HIDDEN_LAYER_NUM][i] = d_sigmoid( u[HIDDEN_LAYER_NUM][i] ) * sum; 	
		}	

		for( n=HIDDEN_LAYER_NUM-1; n > 0; n-- ){

			for( i=0; i < HIDDEN_LAYER; i++ ){
				sum = 0;
				for( k=0; k < HIDDEN_LAYER; k++ ){
				       sum += teach_signal[n+1][k] * weight[n+1][k][i];
				}

				teach_signal[n][i] = d_sigmoid( u[n][i]	) * sum;
			}
		}
			

		//weight overwrite
		
		
		//weight in output_layer 
		
		for( i=0; i < OUTPUT_LAYER; i++ ){
			for( j=0; j < HIDDEN_LAYER; j++ ){
				
				del_weight[LAYER_SUM-1][i][j] = N * teach_signal[LAYER_SUM-1][i] * x[HIDDEN_LAYER][j] + ALPHA * old_del_weight[LAYER_SUM-1][i][j];
				old_del_weight[LAYER_SUM-1][i][j] = del_weight[LAYER_SUM-1][i][j];

				weight[LAYER_SUM-1][i][j] += del_weight[LAYER_SUM-1][i][j];   //output_layer weight update
			}
		}
		

		
		//weight update in hidden_layer
		
		for( n=1; n < LAYER_SUM-1; n++ ){
		       for(i=0; i < HIDDEN_LAYER; i++ ){
			       for( j=0; j < HIDDEN_LAYER; j++ ){
				        del_weight[n][i][j] = N * teach_signal[n][i] * x[n-1][j] + ALPHA * old_del_weight[n][i][j];
				        old_del_weight[n][i][j] = del_weight[n][i][j];
					weight[n][i][j] += del_weight[n][i][j];   	//hidden_layer weight update
			       }
		       }
		}
		

		
		//threshould overwrite
		
		//threshould update in output_layer 	
		for( i=0; i < OUTPUT_LAYER; i++ ){
			
			del_h[LAYER_SUM-1][i] = N * teach_signal[LAYER_SUM-1][i] + ALPHA * old_del_h[LAYER_SUM-1][i];
			old_del_h[LAYER_SUM-1][i] = del_h[LAYER_SUM-1][i];
			h[LAYER_SUM-1][i] += del_h[LAYER_SUM-1][i];
		}


		for( n=1; n < LAYER_SUM-1; n++ ){

			for( i=0; i < HIDDEN_LAYER; i++ ){

				del_h[n][i] = N * teach_signal[n][i] + ALPHA * old_del_h[n][i];
				old_del_h[n][i] = del_h[n][i];
				h[n][i] += del_h[n][i];

			}
		}		   
			
	}


	printf("learned successflly\n");
	//test run
	for( i=0; i < INPUT_LAYER; i++){
		x[0][i] = input_layer[i];
	}
		
	//output by hidden_layer 	
	for( n=1; n < LAYER_SUM-1; n++){
		for( i=0; i < HIDDEN_LAYER; i++){
		       u[n][i] = 0;	
			for( j=0; j < HIDDEN_LAYER; j++){
				u[n][i] += weight[n][i][j] * x[n-1][j];
			}

			x[n][i] = sigmoid( u[n][i] - h[n][i] );


		}
	}
		
	//output by output_layer
	for( i=0; i < OUTPUT_LAYER; i++ ){
		u[LAYER_SUM-1][i] = 0;
		for( j=0; j < HIDDEN_LAYER; j++ ){
			u[LAYER_SUM-1][i] += weight[LAYER_SUM-1][i][j] * x[LAYER_SUM-1][j];

		}
		x[LAYER_SUM-1][i] = sigmoid( u[LAYER_SUM-1][i] - h[LAYER_SUM-1][i] );

		

	}
	
		
	
	for( i=0; i < OUTPUT_LAYER; i++ ){
		printf("%f ",x[LAYER_SUM-1][i] );

	}
	printf("\n");

}

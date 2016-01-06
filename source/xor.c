#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TRIAL       4           //training
#define INPUT       2           //input_data
#define HIDDEN      2          //hidden_data
#define OUTPUT      1           //output_data
#define LEARN       10000        //learn_data
#define ERR_RATE    0.01        //error_rate
#define THETA       0.5          //error_coefficient
#define EPSILON     0.5          //learn_coefficient

#define drand() ((double)(rand() % 15 + 1 ) / 15)


void Enter();
double ABS(double a);
double sigmoid(double z);       //sigmoid_function

int main(){
    unsigned int    i,
                    j,
                    h,
                    o,
                    l,
                    f,
                    e=0,
                    t;

    double  t_input[TRIAL][INPUT],              //teach_input_data

            teach[TRIAL],                      //teach_signal

            input[INPUT],               //input_cells
            hidden[HIDDEN],             //hidden_cells
            output[OUTPUT],             //output_cells

            ih_w[INPUT][HIDDEN],        //input-hidden_weight
            ho_w[HIDDEN][OUTPUT],       //hidden-output_weight

            to_err[OUTPUT],                     //error_rate_of_teach_&_output
            h_err[HIDDEN],                      //error_rate_of_hidden_&_to_err

            h_ts[HIDDEN]={0},                       //hidden_threshould
            o_ts[OUTPUT]={0},                       //output_threshould

            sum;                                    //cells_sum

            srand(time(NULL));

            /*XOR Problem*/
            /*
            0 0  -> 0
            0 1  -> 1
            1 0  -> 1
            1 1  -> 0
            */

            //Training_data
            for(t=0;t<TRIAL;t++){
                if(t==0){
                    t_input[t][0]=0;
                    t_input[t][1]=0;
                    teach[t]=0;
                }
                if(t==1){
                    t_input[t][0]=0;
                    t_input[t][1]=1;
                    teach[t]=1;
                }
                if(t==2){
                    t_input[t][0]=1;
                    t_input[t][1]=0;
                    teach[t]=1;
                }
                if(t==3){
                    t_input[t][0]=1;
                    t_input[t][1]=1;
                    teach[t]=0;
                }
            }

                //Initialize_weight
            printf("Initialize_input-hidden_weight.\n");
                for(i=0;i<INPUT;i++){
                    for(h=0;h<HIDDEN;h++){
                        ih_w[i][h]=drand();
                        printf("ih_w[%d][%d]=%lf\n",i,h,ih_w[i][h]);
                    }
                }

                Enter();
            printf("Initialize_hidden-output_weight.\n");
                for(h=0;h<HIDDEN;h++){
                    for(o=0;o<OUTPUT;o++){
                        ho_w[h][o]=drand();
                        printf("ho_w[%d][%d]=%lf\n",h,o,ho_w[h][o]);
                    }
                }

                Enter();

                for(f=0;f<TRIAL;f++){
                    for(i=0;i<INPUT;i++){
                        input[i]=t_input[f][i];
                        printf("%d input[%d]=%lf\n",f,i,input[i]);
                    }
                    printf("%d teach[%d]=%lf\n\n",f,f,teach[f]);
                }

                        //Nerural nets
                            for(l=1;l<=LEARN;l++){          //learn_loop
                                for(t=0;t<TRIAL;t++){       //training_loop
                                    for(i=0;i<INPUT;i++){
                                        input[i]=t_input[t][i];
                                        printf("input[%d]=%lf\n",i,input[i]);
                                    }
                                    //hidden_data
                                    for(h=0;h<HIDDEN;h++){
                                        sum = 0.0;
                                        for(i=0;i<INPUT;i++){
                                            sum += (ih_w[i][h] * input[i] );
                                        }
                                        sum += h_ts[h];
                                        hidden[h]=sigmoid(sum);
                                    }

                                    Enter();

                                    for(o=0;o<OUTPUT;o++){
                                        sum = 0.0;
                                        for(h=0;h<HIDDEN;h++){
                                            sum += (ho_w[h][o] * hidden[h] );
                                        }
                                        sum += o_ts[o];
                                        output[o]=sigmoid(sum);
                                    }

                                        for(o=0;o<OUTPUT;o++){
                                            to_err[o] = ((teach[t] - output[o]) * output[o] * (1.0 - output[o]));
                                            printf("Trial:%d learn:%d to_err[%d]=%lf\n",t,l,o,ABS(to_err[o]));
                                        }

                                        Enter();

                                        for(h=0;h<HIDDEN;h++){
                                            for(o=0;o<OUTPUT;o++){
                                                h_err[h] = (to_err[o] * ho_w[h][o] * hidden[h] * (1.0 - hidden[h]));
                                                printf("Trial:%d learn:%d h_err[%d]=%lf\n",t,l,h,ABS(h_err[h]));
                                            }
                                        }

                                        for(h=0;h<HIDDEN;h++){
                                            for(i=0;i<INPUT;i++){
                                                ih_w[i][h] += THETA * h_err[h] * input[i];
                                                h_ts[h] += (EPSILON * h_err[h]);
                                            }
                                        }

                                        Enter();

                                            for(o=0;o<OUTPUT;o++){
                                                for(h=0;h<HIDDEN;h++){
                                                    ho_w[h][o] += THETA * to_err[o] * hidden[h];
                                                    o_ts[o] += (EPSILON * to_err[o]);
                                                }
                                            }
                                }
                                to_err[0] = ABS(to_err[0]);
                                if( to_err[0] <= ERR_RATE ){
                                    printf("Break!\n");
                                    break;
                                }

                            }

                    Enter();

                    for(f=0;f<TRIAL;f++){
                        for(i=0;i<INPUT;i++){
                            input[i]=t_input[f][i];
                            printf("%d input[%d]=%lf\n",f,i,input[i]);
                        }
                        printf("%d teach[%d]=%lf\n\n",f,f,teach[f]);
                    }

                    Enter();

                    for(h=0;h<HIDDEN;h++){
                        for(i=0;i<INPUT;i++){
                            printf("ih_w[%d][%d]=%lf\n",i,h,ih_w[i][h]);
                        }
                    }

                    Enter();

                    for(o=0;o<OUTPUT;o++){
                        for(h=0;h<HIDDEN;h++){
                            printf("ho_w[%d][%d]=%lf\n",h,o,ho_w[h][o]);
                        }
                    }
                    Enter();

                    for(h=0;h<HIDDEN;h++){
                        printf("h_ts:%lf\n",h_ts[h]);
                    }

                    Enter();

                    for(o=0;o<OUTPUT;o++){
                        printf("o_ts:%lf\n\n",o_ts[o]);
                    }
        for(;;){
            printf("==================\n");
            for(i=0;i<INPUT;i++){
                printf("input[%d]=",i);
                scanf("%lf",&input[i]);
            }
                    for(h=0;h<HIDDEN;h++){
                        sum = 0.0;
                        for(i=0;i<INPUT;i++){
                            sum += (ih_w[i][h] * input[i] );
                        }
                        sum += h_ts[h];
                        hidden[h]=sigmoid(sum);
                    }

                    Enter();

                    for(o=0;o<OUTPUT;o++){
                        sum = 0.0;
                        for(h=0;h<HIDDEN;h++){
                            sum += (ho_w[h][o] * hidden[h] );
                        }
                        sum += o_ts[o];
                        output[o]=sigmoid(sum);
                    }
                    for(o=0;o<OUTPUT;o++){
                        printf("output=%lf\n\n",output[o]);
                        printf("==================\n");
                    }
        }



}

void Enter(){
    printf("\n");
}
double ABS(double a){
    if(a < 0.0){
        a*=(-1.0);
    }
    return a;
}
double sigmoid(double z){
    double sm;
        sm=(1.0/(1.0 + exp(-z)));
    return sm;
}

#include "graphics.hpp"
#include "../mbp.h"
using namespace genv;
#include <iostream>
using namespace std;

int main()
{
    const int XX=800;
    const int YY=800;
    int reteg[3]={2,25,1};
    MBP mbp(2,reteg,1);
    Trainer tr(&mbp);
    tr.LoadIfPossible(false);
    vector<vector<double> > inp, outp;

    gout.open(XX,YY);
    double **Disp = new REAL*[mbp.Layer()+1];
    for (int i=0; i <= mbp.Layer(); i++) {
        Disp[i]= new REAL[mbp.Unit(i)*(XX*YY)];
    }
    for (int i=0;i<XX;i++)
        for (int j=0;j<YY;j++){
            Disp[0][(i*YY+j)*2]=(i/double(XX))*2-1;
            Disp[0][(i*YY+j)*2+1]=(j/double(YY))*2-1;
        }
    event ev;
    gout << move_to(0,0) << color(128,128,128) << box_to(XX-1,YY-1);
    gout << refresh;

    while(gin >> ev && ev.keycode != key_escape) {
        if (ev.type==ev_mouse) {
            double x=(ev.pos_x/double(XX))*2-1;
            double y=(ev.pos_y/double(YY))*2-1;
            if (ev.button>0) {
                vector<double> p(2);
                p[0]=x;
                p[1]=y;
                inp.push_back(p);
                vector<double> op(1);
                if (ev.button==btn_left)
                    op[0]=0.9;
                if (ev.button==btn_right)
                    op[0]=-0.9;
                outp.push_back(op);
            }
        }
        if (ev.type==ev_key) {
            if (ev.keycode=='c') {
                tr.LoadIfPossible(false);  //reset
                cout << "training reseted" << endl;
            }
            if (ev.keycode=='t') {
                tr.setInput(inp, outp);
                tr.setParams(0, 0, 0.0, -1, 1000, 3, 100, 0);
                tr.Learn();
                tr.LoadIfPossible(true); //ismételt nyomkodás legyen továbbtanulás
                cout <<"err:"<< tr.lastAnaCost() << " ("<<tr.lastMaxCost() <<" max)" << endl;
            }
            if (ev.keycode==' ') {
                mbp.FeedForward(Disp,XX*YY);
                int ll=mbp.Layer();
                for (int i=0;i<XX;i++)
                    for (int j=0;j<YY;j++) {
                         short a=128+100*Disp[ll][i*YY+j];
                         gout << move_to(i,j) << color(a,a,a) << dot;

                    }
            }

        }
        for (int i=0;i<inp.size();i++) {
            short a=128+100*outp[i][0];
            gout << move_to((inp[i][0]+1)/2*XX,(inp[i][1]+1)/2*YY) << color(a,a,a) << box(3,3) << genv::move(-1,-1) << color(128,128,128) << dot;
        }
        gout << refresh;
    }
}

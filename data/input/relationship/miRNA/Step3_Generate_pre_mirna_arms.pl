#!/usr/bin/perl

# This script create a shell to runnning bowtie to align read to human gemone
# use String::Util qw(trim);
sub trim {
    (my $s = $_[0]) =~ s/^\s+|\s+$//g;
    return $s;
}




# for windows
# my $input_armswitch_gff  = "E:\\Macau_workspace\\Alzheimer\\scripts\\input\\miRBase\\v22\\hsa.GRCh38.armswitch.gff3";
# my $output_arms1      = "E:\\Macau_workspace\\Alzheimer\\scripts\\input\\miRBase\\v22\\hsa.GRCh38.arms.CL.style.txt";
# my $output_arms2      = "E:\\Macau_workspace\\Alzheimer\\scripts\\input\\miRBase\\v22\\hsa.GRCh38.arms.miRBase.txt";
# my $output_arms3      = "E:\\Macau_workspace\\Alzheimer\\scripts\\input\\miRBase\\v22\\premirnaID2armsID.txt";



my $input_armswitch_gff  = "/Users/chenliang/Library/CloudStorage/OneDrive-汕头大学/STATIC_DATA/miRBase/output/hsa/gff/hsa.GRCh38.armswitch.gff3";
my $output_arms1      = "/Users/chenliang/Library/CloudStorage/OneDrive-汕头大学/STATIC_DATA/miRBase/output/hsa/hsa.GRCh38.arms.CL.style.txt";
my $output_arms2      = "/Users/chenliang/Library/CloudStorage/OneDrive-汕头大学/STATIC_DATA/miRBase/output/hsa/hsa.GRCh38.arms.miRBase.txt";
my $output_arms3      = "/Users/chenliang/Library/CloudStorage/OneDrive-汕头大学/STATIC_DATA/miRBase/output/hsa/premirnaID2armsID.txt";


my $pre_mirna2arms = get_pre_mirna2arms_name($input_armswitch_gff);
my $pre_mirna2arms_id_name = get_pre_mirna2arms_id_name($input_armswitch_gff);


open OUT_FILE1 , ">".$output_arms1 or die "Sorry can not write file1!\n";
open OUT_FILE2 , ">".$output_arms2 or die "Sorry can not write file1!\n";
open OUT_FILE3 , ">".$output_arms3 or die "Sorry can not write file1!\n";

print OUT_FILE1 "pre_miRNA\tarm_5p\tarm_3p\n";
foreach my $id ( keys %$pre_mirna2arms ){
        my @arms = @{$$pre_mirna2arms{$id}};
        my $num   = @arms ;
        if( $num != 2){
           print  $id." wrong!\n";
           print  "$output_arms1--->$arms[0]\n";
        }
        print OUT_FILE1  "$id\t";
        if( $arms[0] =~ m/-5p$/ ){
           print OUT_FILE1 $arms[0]."\t".$arms[1]."\n";
        }
        if( $arms[0] =~ m/-3p$/ ){
            print OUT_FILE1 $arms[1]."\t".$arms[0]."\n";
        }
}
close OUT_FILE1;

print OUT_FILE2 "pre_miRNA\tarm_5p\tarm_3p\n";
foreach my $id ( keys %$pre_mirna2arms ){
        my @arms = @{$$pre_mirna2arms{$id}};
        my $num   = @arms ;
        my $ids = join(";",@arms);
        if( $num != 2 ){
           print  $id." wrong!\n";
           print  "$output_arms2 --->$arms[0]\n";
        }

        if ($ids =~ m/candidate/) {
            # print "$ids is removed\n";
             next;
        }
        print OUT_FILE2  "$id\t";
        if( $arms[0] =~ m/-5p$/ ){
           print OUT_FILE2 $arms[0]."\t".$arms[1]."\n";
        }
        if( $arms[0] =~ m/-3p$/ ){
            print OUT_FILE2 $arms[1]."\t".$arms[0]."\n";
        }
}
close OUT_FILE2;



print OUT_FILE3 "pre_miRNA\tarm_5p_acc\tarm_5p_name\tarm_3p_acc\tarm_3p_name\n";
foreach my $id ( keys %$pre_mirna2arms ){
        my @arms = @{$$pre_mirna2arms_id_name{$id}};
        my $num   = @arms ;
        my $ids = join(";",@arms);
        if( $num != 2 ){
           print  $id." wrong!\n";
           print  "ids :$ids \n ---> ($num)$arms[0]\n";
        }

        if ($ids =~ m/candidate/) {
             # print "$ids is removed\n";
             next;
        }
        print OUT_FILE3  "$id\t";

        if( $arms[0] =~ m/-5p$/ ){
           print OUT_FILE3 $arms[0]."\t".$arms[1]."\n";
        }
        if( $arms[0] =~ m/-3p$/ ){
            print OUT_FILE3 $arms[1]."\t".$arms[0]."\n";
        }
}
close OUT_FILE3;


print "Done!\n";

sub getName{
    my $IDs  = $_[0];
    if( $IDs =~ /Name=(.*);/ ){
        return $1;
     }
}

sub getDerives_from{
     my $IDs  = trim($_[0]);
     if( $IDs =~ /Derives_from=(.*)/ ){
        return $1;
     }
}

sub getID{
     my $IDs = $_[0];
     if( $IDs =~ /ID=(.*);Alias/ ){
         return $1;
     }
}


sub getAlias{
     my $IDs = $_[0];
     if( $IDs =~ /Alias=(.*);Name/ ){
         return $1;
     }
}


sub get_pre_mirna2arms_name{
    my $file = $_[0];
    my @one_arm_mirna = ();
    open IN_FILE   , "<".$file  or die "Sorry can not open file!\n";
    my %pre_mirna2arms = ();
    while(<IN_FILE>){
       my $line = trim($_);
       if( substr($line,0, 1) eq "#" ){
          next;
       }
       my @temp = split /\t/,$line;
       my $IDs  = $temp[8];
       my $type = $temp[2];

       if( $type eq "miRNA_primary_transcript" ){
#          $pre_mirna_id = getID($IDs);
          $pre_mirna_id = getAlias($IDs);
          my @arms = ();
          $pre_mirna2arms{$pre_mirna_id} = \@arms;
       } else{
          $mirna_name = getName($IDs);
          $derives_from = getDerives_from($IDs);
          if( defined $pre_mirna2arms{$derives_from} ){
              # push $pre_mirna2arms{$derives_from},$mirna_name;
              push @{$pre_mirna2arms{$derives_from}},$mirna_name;
             # print $derives_from."\t".$mirna_name."\n";
          } else{
             print "Sorry, pre-mirna is not apear before mirna!\n ";
          }
       }
    }
    return  \%pre_mirna2arms;
}



sub get_pre_mirna2arms_id_name{
    my $file = $_[0];
    my @one_arm_mirna = ();
    open IN_FILE   , "<".$file  or die "Sorry can not open file!\n";
    my %pre_mirna2arms = ();
    while(<IN_FILE>){
       my $line = trim($_);
       if( substr($line,0, 1) eq "#" ){
          next;
       }
       my @temp = split /\t/,$line;
       my $IDs  = $temp[8];
       my $type = $temp[2];

       if( $type eq "miRNA_primary_transcript" ){
          $pre_mirna_id = getID($IDs);
          my @arms = ();
          $pre_mirna2arms{$pre_mirna_id} = \@arms;
       } else{

          # my rename or candidate no ID and Alias
          $mirna_id = getAlias($IDs);

          $mirna_name = getName($IDs);
          $derives_from = getDerives_from($IDs);
          if( defined $pre_mirna2arms{$derives_from} ){

             # push $pre_mirna2arms{$derives_from},$mirna_id."\t".$mirna_name;
             push @{$pre_mirna2arms{$derives_from}},$mirna_id."\t".$mirna_name;
             # print $derives_from."\t".$mirna_name."\n";
          } else{
             print "Sorry, pre-mirna is not apear before mirna!\n ";
          }
       }
    }
    return  \%pre_mirna2arms;
}